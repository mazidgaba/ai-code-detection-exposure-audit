"""Controlled experiments for the paper.

Every experiment shares one reduced-but-fixed training configuration so the
runs are comparable to each other. They are deliberately NOT comparable to the
headline branch-B numbers, which used the larger cpu.yaml settings; the point
here is the delta between conditions, not the absolute level.

  E1  split methodology     random row split vs problem-wise grouping
  E2  label granularity     binary vs ternary vs four-class, scored on hybrid
  E3  formatter robustness  evasion, and augmentation as the defence
  E4  feature ablation      tf-idf / stylometry / AST and their combinations

Results checkpoint to eval/reports/experiments.json after each run, so a
partial sweep is still usable.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from aicd import config as C
from aicd.data.splits import SLICES
from aicd.models.formatter_ablation import normalize_whitespace

# Fixed across every experiment below.
N_TRAIN = 10000
N_EVAL = 2000
TFIDF_FEATURES = 2500
N_TREES = 70
MAX_DEPTH = 5
SEED = 20260818


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                           max_features=TFIDF_FEATURES, sublinear_tf=True,
                           min_df=3, lowercase=False)


def make_clf(n_classes: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=N_TREES, max_depth=MAX_DEPTH, learning_rate=0.15,
        subsample=0.85, colsample_bytree=0.7, n_jobs=-1,
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        tree_method="hist", random_state=SEED, eval_metric="mlogloss",
    )


def fit_eval(train_df, train_dense, evals: dict, n_classes: int,
             use_tfidf=True, use_dense=True, dense_cols=None):
    """Fit on train_df, return {name: (y_true, proba)} for each eval frame."""
    vec = make_vectorizer() if use_tfidf else None
    parts = []
    if use_tfidf:
        parts.append(vec.fit_transform(train_df["code"]))
    if use_dense:
        cols = dense_cols if dense_cols is not None else list(train_dense.columns)
        parts.append(sp.csr_matrix(train_dense[cols].astype(np.float32).values))
    else:
        cols = []
    Xtr = sp.hstack(parts, format="csr") if len(parts) > 1 else parts[0]
    ytr = train_df["label"].to_numpy()

    clf = make_clf(n_classes)
    clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr), verbose=False)

    out = {}
    for name, (edf, edense) in evals.items():
        p = []
        if use_tfidf:
            p.append(vec.transform(edf["code"]))
        if use_dense:
            p.append(sp.csr_matrix(edense[cols].astype(np.float32).values))
        Xe = sp.hstack(p, format="csr") if len(p) > 1 else p[0]
        proba = clf.predict_proba(Xe)
        if proba.ndim == 1 or proba.shape[1] == 1:
            proba = np.column_stack([1 - proba.ravel(), proba.ravel()])
        out[name] = (edf["label"].to_numpy(), proba)
    return out, clf


def macro_f1(y, proba) -> float:
    return float(f1_score(y, proba.argmax(1), average="macro", zero_division=0))


def load_all(cfg):
    d = C.ROOT / cfg.data.cache_dir
    df = pd.read_parquet(d / "splits.parquet")
    dense = pd.read_parquet(d / "dense_features.parquet")
    dense.index = df.index
    return df, dense


def sample(df, dense, n, seed=SEED):
    if len(df) <= n:
        return df, dense
    idx = df.sample(n=n, random_state=seed).index
    return df.loc[idx], dense.loc[idx]


def eval_frames(df, dense, n=N_EVAL):
    out = {}
    for s in SLICES:
        m = df["slice"] == s
        if not m.any():
            continue
        sd, sden = sample(df[m], dense[m], n)
        out[s] = (sd, sden)
    return out


# ---------------------------------------------------------------- E1
def e1_split_methodology(df, dense, cfg) -> dict:
    """Does a random row split inflate the score relative to grouping by problem?

    Oedingen et al. reported roughly +4 accuracy points of illusion from a
    random split. Both arms here train on the same number of rows drawn from
    the same pool; only the train/test boundary differs.
    """
    log("E1: split methodology")
    pool = df[df["slice"].isin(["train", "val", "s1_in_distribution"])]
    pool_dense = dense.loc[pool.index]
    res = {}

    # Arm A -- problem-wise grouping (correct).
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_i, te_i = next(gss.split(pool, groups=pool["problem_id"]))
    tr, te = pool.iloc[tr_i], pool.iloc[te_i]
    trd, ted = pool_dense.iloc[tr_i], pool_dense.iloc[te_i]
    tr, trd = sample(tr, trd, N_TRAIN)
    te, ted = sample(te, ted, N_EVAL)
    o, _ = fit_eval(tr, trd, {"test": (te, ted)}, 4)
    res["problem_wise"] = {"macro_f1": macro_f1(*o["test"]), "n_train": len(tr), "n_test": len(te)}
    log(f"  problem-wise: {res['problem_wise']['macro_f1']:.4f}")

    # Arm B -- random row split (leaks near-duplicate solutions).
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(pool))
    cut = int(len(pool) * 0.8)
    tr, te = pool.iloc[perm[:cut]], pool.iloc[perm[cut:]]
    trd, ted = pool_dense.iloc[perm[:cut]], pool_dense.iloc[perm[cut:]]
    tr, trd = sample(tr, trd, N_TRAIN)
    te, ted = sample(te, ted, N_EVAL)
    o, _ = fit_eval(tr, trd, {"test": (te, ted)}, 4)
    res["random_row"] = {"macro_f1": macro_f1(*o["test"]), "n_train": len(tr), "n_test": len(te)}
    log(f"  random-row  : {res['random_row']['macro_f1']:.4f}")

    res["inflation"] = res["random_row"]["macro_f1"] - res["problem_wise"]["macro_f1"]
    log(f"  inflation   : {res['inflation']:+.4f}")
    return res


# ---------------------------------------------------------------- E2
def e2_label_granularity(df, dense, cfg) -> dict:
    """What does binary training cost on hybrid code?

    CoDet-M4 reported hybrid F1 of 39.4 under binary training, recovering to
    86.1 once hybrid became its own class. The binary arm here follows their
    protocol: train on human vs machine only, then score hybrid samples as
    correct whenever the model calls them non-human.
    """
    log("E2: label granularity")
    tr_all = df[df["split"] == "train"]
    ev = df[df["slice"] == "s1_in_distribution"]
    ev_h = ev[ev["label"] == 2]
    res = {}

    # Arm A -- binary, hybrid never seen in training.
    tr = tr_all[tr_all["label"].isin([0, 1])]
    trd = dense.loc[tr.index]
    tr, trd = sample(tr, trd, N_TRAIN)
    ev_s, ev_sd = sample(ev, dense.loc[ev.index], N_EVAL)
    evh_s, evh_sd = sample(ev_h, dense.loc[ev_h.index], N_EVAL)
    o, _ = fit_eval(tr, trd, {"hybrid": (evh_s, evh_sd), "all": (ev_s, ev_sd)}, 2)
    yh, ph = o["hybrid"]
    hybrid_recall_bin = float((ph.argmax(1) != 0).mean())
    yb = (ev_s["label"].to_numpy() != 0).astype(int)
    res["binary"] = {
        "hybrid_recall": hybrid_recall_bin,
        "binary_macro_f1": float(f1_score(yb, o["all"][1].argmax(1), average="macro", zero_division=0)),
    }
    log(f"  binary  : hybrid recall {hybrid_recall_bin:.4f}")

    # Arm B -- ternary: human / machine+adversarial / hybrid.
    tr = tr_all.copy()
    tr["label"] = tr["label"].map({0: 0, 1: 1, 3: 1, 2: 2})
    trd = dense.loc[tr.index]
    tr, trd = sample(tr, trd, N_TRAIN)
    ev3 = ev_s.copy()
    ev3["label"] = ev3["label"].map({0: 0, 1: 1, 3: 1, 2: 2})
    o, _ = fit_eval(tr, trd, {"all": (ev3, ev_sd)}, 3)
    y3, p3 = o["all"]
    f_h3 = float(f1_score(y3, p3.argmax(1), average=None, labels=[2], zero_division=0)[0])
    res["ternary"] = {"hybrid_f1": f_h3, "macro_f1": macro_f1(y3, p3),
                      "hybrid_recall": float(recall_score(y3, p3.argmax(1), labels=[2],
                                                          average="macro", zero_division=0))}
    log(f"  ternary : hybrid F1 {f_h3:.4f}")

    # Arm C -- four-class.
    tr, trd = sample(tr_all, dense.loc[tr_all.index], N_TRAIN)
    o, _ = fit_eval(tr, trd, {"all": (ev_s, ev_sd)}, 4)
    y4, p4 = o["all"]
    f_h4 = float(f1_score(y4, p4.argmax(1), average=None, labels=[2], zero_division=0)[0])
    res["four_class"] = {"hybrid_f1": f_h4, "macro_f1": macro_f1(y4, p4),
                         "hybrid_recall": float(recall_score(y4, p4.argmax(1), labels=[2],
                                                             average="macro", zero_division=0))}
    log(f"  4-class : hybrid F1 {f_h4:.4f}")
    return res


# ---------------------------------------------------------------- E3
def e3_formatter(df, dense, cfg) -> dict:
    """Formatter evasion, and augmentation as the defence.

    Arm A is a plain model meeting normalized input it never trained on.
    Arm B trains on the union of raw and normalized copies. If augmentation
    works, the evasion gap closes without costing much on raw input.
    """
    log("E3: formatter evasion and augmentation")
    from aicd.features.build import dense_features

    tr_all = df[df["split"] == "train"]
    tr, trd = sample(tr_all, dense.loc[tr_all.index], N_TRAIN)

    evs = {}
    for s in ["s1_in_distribution", "s2_unseen_generator"]:
        m = df["slice"] == s
        sd, sden = sample(df[m], dense[m], 2000)
        evs[s] = (sd, sden)
        fmt = sd.copy()
        fmt["code"] = fmt["code"].map(normalize_whitespace)
        evs[s + "_FMT"] = (fmt, dense_features(fmt).reindex(columns=dense.columns, fill_value=0.0))

    res = {}
    o, _ = fit_eval(tr, trd, evs, 4)
    res["plain"] = {k: macro_f1(*v) for k, v in o.items()}
    log(f"  plain      raw={res['plain']['s1_in_distribution']:.4f} "
        f"fmt={res['plain']['s1_in_distribution_FMT']:.4f}")

    # Augmented: raw rows plus a normalized copy of each.
    aug = tr.copy()
    aug["code"] = aug["code"].map(normalize_whitespace)
    aug_dense = dense_features(aug).reindex(columns=dense.columns, fill_value=0.0)
    aug_dense.index = aug.index
    tr_aug = pd.concat([tr, aug], ignore_index=True)
    trd_aug = pd.concat([trd, aug_dense], ignore_index=True)
    o, _ = fit_eval(tr_aug, trd_aug, evs, 4)
    res["augmented"] = {k: macro_f1(*v) for k, v in o.items()}
    log(f"  augmented  raw={res['augmented']['s1_in_distribution']:.4f} "
        f"fmt={res['augmented']['s1_in_distribution_FMT']:.4f}")

    res["gap_plain"] = res["plain"]["s1_in_distribution"] - res["plain"]["s1_in_distribution_FMT"]
    res["gap_augmented"] = res["augmented"]["s1_in_distribution"] - res["augmented"]["s1_in_distribution_FMT"]
    log(f"  evasion gap {res['gap_plain']:+.4f} -> {res['gap_augmented']:+.4f}")
    return res


# ---------------------------------------------------------------- E4
def e4_feature_ablation(df, dense, cfg) -> dict:
    """Which representation carries the signal, and which one survives shift?"""
    log("E4: feature ablation")
    tr_all = df[df["split"] == "train"]
    tr, trd = sample(tr_all, dense.loc[tr_all.index], N_TRAIN)
    evs = eval_frames(df, dense, 2000)

    sty_cols = [c for c in dense.columns if not c.startswith("ast_")]
    ast_cols = [c for c in dense.columns if c.startswith("ast_")]

    configs = {
        "tfidf_only":       dict(use_tfidf=True,  use_dense=False, dense_cols=None),
        "stylometry_only":  dict(use_tfidf=False, use_dense=True,  dense_cols=sty_cols),
        "ast_only":         dict(use_tfidf=False, use_dense=True,  dense_cols=ast_cols),
        "stylometry_ast":   dict(use_tfidf=False, use_dense=True,  dense_cols=list(dense.columns)),
        "all":              dict(use_tfidf=True,  use_dense=True,  dense_cols=list(dense.columns)),
    }
    res = {}
    for name, kw in configs.items():
        o, _ = fit_eval(tr, trd, evs, 4, **kw)
        res[name] = {k: macro_f1(*v) for k, v in o.items()}
        log(f"  {name:16s} s1={res[name]['s1_in_distribution']:.4f} "
            f"s5={res[name].get('s5_compound', float('nan')):.4f}")
    return res


EXPERIMENTS = {"e1": e1_split_methodology, "e2": e2_label_granularity,
               "e3": e3_formatter, "e4": e4_feature_ablation}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu.yaml")
    ap.add_argument("--only", nargs="+", default=None, choices=list(EXPERIMENTS))
    args = ap.parse_args()
    cfg = C.load(args.config)

    dest = C.reports(cfg) / "experiments.json"
    all_res = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}

    df, dense = load_all(cfg)
    log(f"corpus {len(df):,} rows, dense {dense.shape}")
    log(f"protocol: n_train={N_TRAIN:,} n_eval={N_EVAL:,} "
        f"tfidf={TFIDF_FEATURES} trees={N_TREES} depth={MAX_DEPTH}")

    for key in (args.only or list(EXPERIMENTS)):
        t = time.time()
        all_res[key] = EXPERIMENTS[key](df, dense, cfg)
        all_res[key]["_elapsed_min"] = round((time.time() - t) / 60, 2)
        all_res["_protocol"] = {"n_train": N_TRAIN, "n_eval": N_EVAL,
                                "tfidf_features": TFIDF_FEATURES,
                                "n_trees": N_TREES, "max_depth": MAX_DEPTH, "seed": SEED}
        dest.write_text(json.dumps(all_res, indent=2), encoding="utf-8")
        log(f"{key} done in {all_res[key]['_elapsed_min']} min -> {dest.name}")


if __name__ == "__main__":
    main()
