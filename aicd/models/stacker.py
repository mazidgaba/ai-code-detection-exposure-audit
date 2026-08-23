"""Fusion: multinomial logistic regression over the three branches.

Fitted on the VALIDATION split only. Branch probabilities on the training split
are overfit -- the branches memorized those rows -- so a stacker fitted there
learns to trust whichever branch overfits hardest.

Simple beats fancy: the SemEval-2026 field found logit-level stacking
outperformed the more elaborate ensembles teams tried.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from aicd import config as C
from aicd.data.splits import SLICES
from aicd.eval import metrics as M


def assemble(cfg, slice_name: str, n_rows: int) -> np.ndarray | None:
    """Stack [branch_a(4) | branch_b(4) | branch_c(1)] for one slice.

    A missing branch contributes a uniform/neutral block rather than removing
    the slice, so the stacker still trains when a branch has not been run.
    """
    art = C.artifacts(cfg)
    blocks = []

    for tag, width in (("a", 4), ("b", 4)):
        p = art / f"proba_{tag}_{slice_name}.npy"
        if p.exists():
            arr = np.load(p)
            if len(arr) == n_rows:
                blocks.append(arr)
                continue
            print(f"  [warn] proba_{tag}_{slice_name} has {len(arr)} rows, expected {n_rows}")
        blocks.append(np.full((n_rows, width), 0.25, dtype=np.float32))

    pc = art / f"score_c_{slice_name}.npy"
    if pc.exists():
        sc = np.load(pc)
        col = sc.reshape(-1, 1) if len(sc) == n_rows else np.zeros((n_rows, 1), np.float32)
    else:
        col = np.zeros((n_rows, 1), np.float32)
    blocks.append(col.astype(np.float32))

    return np.hstack(blocks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    args = ap.parse_args()
    cfg = C.load(args.config)

    df = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet")

    val = df[df["slice"] == "val"]
    Xv = assemble(cfg, "val", len(val))
    yv = val["label"].to_numpy()
    print(f"[stacker] fitting on val only: {Xv.shape}")

    clf = LogisticRegression(
        max_iter=2000, multi_class="multinomial",
        class_weight="balanced", C=1.0, random_state=cfg.project.seed,
    )
    clf.fit(Xv, yv)

    with open(C.artifacts(cfg) / "stacker.pkl", "wb") as f:
        pickle.dump(clf, f)

    proba_by_slice = {}
    for s in SLICES:
        m = df["slice"] == s
        if not m.any():
            continue
        X = assemble(cfg, s, int(m.sum()))
        p = clf.predict_proba(X)
        np.save(C.artifacts(cfg) / f"proba_stack_{s}.npy", p)
        proba_by_slice[s] = (df.loc[m, "label"].to_numpy(), p)

    res = M.evaluate_all(df, proba_by_slice, "stacker", C.reports(cfg))
    M.print_table(res)

    names = [f"a_{i}" for i in range(4)] + [f"b_{i}" for i in range(4)] + ["c_curv"]
    print("\nstacker coefficients (per class, higher = pushes toward that class)")
    print(f"{'feature':10s}" + "".join(f"{n:>12s}" for n in
                                       ["human", "machine", "hybrid", "adversarial"]))
    for j, n in enumerate(names):
        print(f"{n:10s}" + "".join(f"{clf.coef_[k][j]:>12.3f}" for k in range(clf.coef_.shape[0])))


if __name__ == "__main__":
    main()
