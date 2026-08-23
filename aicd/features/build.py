"""Assemble the dense feature matrix (stylometry + AST) once and cache it.

Feature selection rule from CoDet-M4: drop any column missing (or constant-zero)
in more than `max_missing_fraction` of training rows. With per-node-type AST
densities this removes several hundred sparse language-specific columns.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from aicd import config as C
from aicd.features import ast_feats, stylometry


def dense_features(df: pd.DataFrame) -> pd.DataFrame:
    sty = stylometry.extract(df)
    ast = ast_feats.extract(df)
    return pd.concat([sty, ast], axis=1)


def select_columns(X: pd.DataFrame, max_missing: float, per_language=None) -> list[str]:
    """Keep features that actually carry information.

    CoDet-M4 dropped features missing in >20% of rows, but they ran a
    3-language corpus. Applied globally to 9 languages that rule deletes every
    language-specific AST node type -- a node type present in every Python
    file still appears in only ~38% of a mixed corpus. So the threshold is
    evaluated PER LANGUAGE and a feature is kept if it clears the bar in any
    single language.
    """
    keep = set()
    thresh = 1.0 - max_missing
    if per_language is None:
        nz = (X != 0).mean(axis=0)
        keep = set(X.columns[nz >= thresh])
    else:
        for lang in pd.Series(per_language).unique():
            m = (pd.Series(per_language).to_numpy() == lang)
            if m.sum() < 50:
                continue
            nz = (X.loc[m] != 0).mean(axis=0)
            keep |= set(X.columns[nz >= thresh])
    return sorted(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--input", default="splits.parquet")
    args = ap.parse_args()
    cfg = C.load(args.config)
    d = C.ROOT / cfg.data.cache_dir

    df = pd.read_parquet(d / args.input)
    print(f"[features] building dense features for {len(df):,} rows...")
    X = dense_features(df)
    print(f"  raw dense width: {X.shape[1]}")

    X.to_parquet(d / "dense_features_raw.parquet", index=False)
    print("  cached raw matrix -> dense_features_raw.parquet")

    train_mask = df["split"] == "train"
    cols = select_columns(X.loc[train_mask], cfg.features.max_missing_fraction,
                          per_language=df.loc[train_mask, "language"])
    print(f"  after per-language >{int(cfg.features.max_missing_fraction*100)}% missing filter: {len(cols)}")

    X = X[cols].astype(np.float32)
    X.to_parquet(d / "dense_features.parquet", index=False)
    with open(C.artifacts(cfg) / "dense_columns.json", "w", encoding="utf-8") as f:
        json.dump(cols, f, indent=1)
    print(f"[done] dense_features.parquet {X.shape}")


if __name__ == "__main__":
    main()
