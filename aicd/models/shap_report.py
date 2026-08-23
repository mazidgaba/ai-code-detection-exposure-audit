"""SHAP for branch B's dense features.

Expected top signals, per CoDet-M4: empty-line count and AST depth. If those
don't rank, the feature extraction is broken -- treat it as a failing check.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from aicd import config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--sample", type=int, default=2000)
    args = ap.parse_args()
    cfg = C.load(args.config)
    art = C.artifacts(cfg)

    import shap
    from xgboost import XGBClassifier

    with open(art / "xgb_branch_b.pkl", "rb") as f:
        bundle = pickle.load(f)
    dcols = bundle["dense_cols"]

    d = C.ROOT / cfg.data.cache_dir
    df = pd.read_parquet(d / "splits.parquet")
    dense = pd.read_parquet(d / "dense_features.parquet")
    tr = df["split"] == "train"
    Xd = dense.loc[tr, dcols]
    yd = df.loc[tr, "label"]
    n = min(args.sample, len(Xd))
    idx = np.random.default_rng(cfg.project.seed).choice(len(Xd), n, replace=False)

    # Dense-only surrogate: SHAP over 20k sparse tf-idf columns is not
    # interpretable anyway, and the question here is which *human-readable*
    # features carry signal.
    sur = XGBClassifier(n_estimators=250, max_depth=6, tree_method="hist",
                        objective="multi:softprob", num_class=4,
                        random_state=cfg.project.seed)
    sur.fit(Xd.iloc[idx], yd.iloc[idx])

    expl = shap.TreeExplainer(sur)
    sv = expl.shap_values(Xd.iloc[idx])
    # Shape depends on the SHAP version: a list of per-class (n, f) arrays, or
    # a single (n, f, c) array. Reduce every axis EXCEPT the feature axis --
    # collapsing the wrong one silently yields one value per class.
    a = np.abs(np.asarray(sv))
    if a.ndim == 3:
        if a.shape[1] == len(dcols):        # (n, f, c)
            arr = a.mean(axis=(0, 2))
        else:                                # (c, n, f)
            arr = a.mean(axis=(0, 1))
    else:                                    # (n, f)
        arr = a.mean(axis=0)
    assert arr.shape[0] == len(dcols), f"shap reduced to {arr.shape}, expected {len(dcols)}"
    order = np.argsort(arr)[::-1][:20]

    print(f"\ntop 20 dense features by mean |SHAP| (n={n})")
    print(f"{'rank':>4s}  {'feature':34s} {'mean|shap|':>10s}")
    top = []
    for r, i in enumerate(order, 1):
        print(f"{r:>4d}  {dcols[i]:34s} {arr[i]:>10.5f}")
        top.append(dcols[i])

    expected = {"n_empty_lines", "r_empty_lines", "ast_max_depth", "ast_mean_depth"}
    hit = expected & set(top)
    print(f"\npublished-signal check: {sorted(hit) if hit else 'NONE FOUND'}")
    if not hit:
        print("  [warn] neither empty-line nor AST-depth features ranked top-20.")
        print("         CoDet-M4 found both dominant. Inspect feature extraction.")

    pd.DataFrame({"feature": [dcols[i] for i in order],
                  "mean_abs_shap": arr[order]}).to_csv(
        C.reports(cfg) / "shap_top20.csv", index=False)


if __name__ == "__main__":
    main()
