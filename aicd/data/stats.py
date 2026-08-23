"""Phase 01d: corpus composition report. Warns on thin cells."""
from __future__ import annotations

import argparse

import pandas as pd

from aicd import config as C
from aicd.config import LABEL_NAMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--input", default="filtered.parquet")
    args = ap.parse_args()
    cfg = C.load(args.config)
    df = pd.read_parquet(C.ROOT / cfg.data.cache_dir / args.input)
    df["label_name"] = df["label"].map(dict(enumerate(LABEL_NAMES)))

    print(f"total rows: {len(df):,}\n")
    for dim in ("label_name", "language", "domain", "model_family"):
        print(f"--- {dim} ---")
        print(df[dim].value_counts().to_string())
        print()

    print("--- label x language ---")
    print(pd.crosstab(df["language"], df["label_name"]).to_string())
    print("\n--- label x domain ---")
    print(pd.crosstab(df["domain"], df["label_name"]).to_string())

    thin = []
    for keys, grp in df.groupby(["language", "domain", "label_name"]):
        if len(grp) < 100:
            thin.append((*keys, len(grp)))
    if thin:
        print(f"\n[warn] {len(thin)} cells under 100 samples (first 20):")
        for t in thin[:20]:
            print(f"  {t[0]:12s} {t[1]:18s} {t[2]:14s} {t[3]:>5,}")


if __name__ == "__main__":
    main()
