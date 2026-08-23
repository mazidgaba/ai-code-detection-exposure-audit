"""Branch B: TF-IDF char n-grams + stylometric + AST features -> XGBoost.

This is the robustness anchor and it stays in the ensemble permanently.
AICD Bench found classical models over TF-IDF beat all six neural encoders
under genuine distribution shift, and Oedingen et al. found XGB+TF-IDF was
the best-calibrated model they tested.
"""
from __future__ import annotations

import argparse
import gc
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.utils.class_weight import compute_sample_weight

from aicd import config as C
from aicd.data.splits import SLICES
from aicd.eval import metrics as M
from aicd.features import tfidf as T


def load(cfg):
    d = C.ROOT / cfg.data.cache_dir
    df = pd.read_parquet(d / "splits.parquet")
    dense = pd.read_parquet(d / "dense_features.parquet")
    assert len(df) == len(dense), "dense_features is stale, rerun features/build.py"
    return df, dense


def fit(cfg, df, dense, save: bool = True):
    from xgboost import XGBClassifier

    tr = df["split"] == "train"
    vec = T.build(cfg)
    print(f"[xgb] fitting TF-IDF on {int(tr.sum()):,} training rows...")
    Xtr_t = vec.fit_transform(df.loc[tr, "code"])
    print(f"  tfidf width: {Xtr_t.shape[1]}  nnz: {Xtr_t.nnz:,}")

    # float64 is wasted here: TF-IDF weights carry nothing like 15 significant
    # digits, and the matrix is the single largest object in the process. On a
    # 200k-row corpus at 20k features this is the difference between fitting in
    # RAM and being killed by the OOM reaper.
    Xtr_t = Xtr_t.astype(np.float32)

    dcols = list(dense.columns)
    dense_tr = sp.csr_matrix(dense.loc[tr].to_numpy(dtype=np.float32))
    ytr = df.loc[tr, "label"].to_numpy()

    Xtr = sp.hstack([Xtr_t, dense_tr], format="csr")
    # hstack has already copied both inputs. Holding them alive through the fit
    # doubles peak memory for no reason.
    del Xtr_t, dense_tr
    gc.collect()
    print(f"  design matrix: {Xtr.shape} nnz={Xtr.nnz:,} "
          f"~{Xtr.data.nbytes / 1e9 + Xtr.indices.nbytes / 1e9:.2f} GB")

    clf = XGBClassifier(
        n_estimators=cfg.xgb.n_estimators,
        max_depth=cfg.xgb.max_depth,
        learning_rate=cfg.xgb.learning_rate,
        subsample=cfg.xgb.subsample,
        colsample_bytree=cfg.xgb.colsample_bytree,
        n_jobs=cfg.xgb.n_jobs,
        objective="multi:softprob",
        num_class=4,
        tree_method="hist",
        random_state=cfg.project.seed,
        eval_metric="mlogloss",
    )
    sw = compute_sample_weight("balanced", ytr)
    print(f"[xgb] training {cfg.xgb.n_estimators} trees on {Xtr.shape}...")
    clf.fit(Xtr, ytr, sample_weight=sw, verbose=False)

    if save:
        art = C.artifacts(cfg)
        with open(art / "xgb_branch_b.pkl", "wb") as f:
            pickle.dump({"vec": vec, "clf": clf, "dense_cols": dcols}, f)
        print(f"[xgb] saved -> {art/'xgb_branch_b.pkl'}")
    return vec, clf, dcols


def align_dense(dense_df: pd.DataFrame, dense_cols: list[str]) -> pd.DataFrame:
    """Reindex to the training column set, filling absent columns with zero.

    A subsample of rows will not contain every AST node type the model was
    trained on -- a Python-only batch has no `ast_d_method_invocation`. Absent
    means density zero, which is exactly what reindex fills in.
    """
    return dense_df.reindex(columns=dense_cols, fill_value=0.0).astype(np.float32)


def predict(vec, clf, dense_cols, codes, dense_df) -> np.ndarray:
    X = sp.hstack(
        [vec.transform(codes), sp.csr_matrix(align_dense(dense_df, dense_cols).values)],
        format="csr",
    )
    return clf.predict_proba(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    args = ap.parse_args()
    cfg = C.load(args.config)

    df, dense = load(cfg)
    vec, clf, dcols = fit(cfg, df, dense)

    proba_by_slice = {}
    for s in ["val"] + SLICES:
        m = df["slice"] == s
        if not m.any():
            continue
        p = predict(vec, clf, dcols, df.loc[m, "code"], dense.loc[m])
        proba_by_slice[s] = (df.loc[m, "label"].to_numpy(), p)
        np.save(C.artifacts(cfg) / f"proba_b_{s}.npy", p)

    res = M.evaluate_all(df, proba_by_slice, "branch_b_xgb", C.reports(cfg))
    M.print_table(res)


if __name__ == "__main__":
    main()
