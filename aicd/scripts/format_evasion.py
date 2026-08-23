"""How much of branch B's signal survives a code formatter?

Oedingen et al. asked this by retraining on formatted data and lost ~4 points.
The more actionable production question is the evasion one: if someone runs a
formatter over AI-generated code before submitting it, does the DEPLOYED model
still catch it? That needs no retraining -- score the existing model on
normalized inputs and measure the drop.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from aicd import config as C
from aicd.data.splits import SLICES
from aicd.features.build import dense_features
from aicd.models.formatter_ablation import normalize_whitespace
from aicd.models.xgb import predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu.yaml")
    ap.add_argument("--max-per-slice", type=int, default=3000)
    args = ap.parse_args()
    cfg = C.load(args.config)

    with open(C.artifacts(cfg) / "xgb_branch_b.pkl", "rb") as f:
        b = pickle.load(f)
    df = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet")

    print(f"{'slice':22s} {'n':>6s} {'F1 raw':>8s} {'F1 fmt':>8s} {'delta':>8s} "
          f"{'AUC raw':>8s} {'AUC fmt':>8s}")
    print("-" * 72)
    rows = []
    for s in SLICES:
        sub = df[df["slice"] == s]
        if sub.empty:
            continue
        if len(sub) > args.max_per_slice:
            sub = sub.sample(n=args.max_per_slice, random_state=cfg.project.seed)
        y = sub["label"].to_numpy()
        yb = (y != 0).astype(int)

        p_raw = predict(b["vec"], b["clf"], b["dense_cols"], sub["code"], dense_features(sub))
        fmt = sub.copy()
        fmt["code"] = fmt["code"].map(normalize_whitespace)
        p_fmt = predict(b["vec"], b["clf"], b["dense_cols"], fmt["code"], dense_features(fmt))

        f_raw = f1_score(y, p_raw.argmax(1), average="macro", zero_division=0)
        f_fmt = f1_score(y, p_fmt.argmax(1), average="macro", zero_division=0)
        a_raw = roc_auc_score(yb, 1 - p_raw[:, 0]) if len(np.unique(yb)) == 2 else float("nan")
        a_fmt = roc_auc_score(yb, 1 - p_fmt[:, 0]) if len(np.unique(yb)) == 2 else float("nan")

        print(f"{s:22s} {len(sub):>6,} {f_raw:>8.4f} {f_fmt:>8.4f} {f_fmt - f_raw:>+8.4f} "
              f"{a_raw:>8.4f} {a_fmt:>8.4f}")
        rows.append({"slice": s, "f1_raw": f_raw, "f1_formatted": f_fmt,
                     "delta": f_fmt - f_raw, "auc_raw": a_raw, "auc_formatted": a_fmt})

    pd.DataFrame(rows).to_csv(C.reports(cfg) / "format_evasion.csv", index=False)
    d = np.mean([r["delta"] for r in rows])
    print(f"\nmean macro-F1 delta: {d:+.4f}")
    print("Oedingen et al. lost ~4-8 points to formatting. A much larger drop")
    print("here would mean the detector is mostly a whitespace detector.")


if __name__ == "__main__":
    main()
