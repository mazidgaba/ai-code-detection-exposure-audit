"""Phase 02: the OOD split matrix.

Five evaluation slices, all disjoint from train by problem_id:

  s1_in_distribution  seen language, domain and generator family
  s2_unseen_generator two model families held out of training entirely
  s3_unseen_language  train py/java/cpp/c, test go/javascript
  s4_unseen_domain    held-out SOURCES (class-level, inline, research code)
  s5_compound         unseen language AND unseen domain

Splitting is by problem_id, never by row. A random row split puts near-identical
solutions to the same problem on both sides of the boundary and inflates every
metric by roughly four points (Oedingen et al. measured exactly this).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from aicd import config as C


def assign(df: pd.DataFrame, cfg) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.project.seed)
    df = df.copy()

    fams = set(cfg.splits.holdout_families)
    langs = set(cfg.splits.holdout_languages)
    srcs = set(cfg.splits.holdout_sources)

    is_fam = df["model_family"].isin(fams)
    is_lang = df["language"].isin(langs)
    is_dom = df["source"].isin(srcs)

    # A sample is eligible for training only if it is clean on every axis.
    trainable = ~(is_fam | is_lang | is_dom)

    slice_of = pd.Series("unused", index=df.index, dtype=object)
    slice_of[is_lang & is_dom] = "s5_compound"
    slice_of[is_lang & ~is_dom] = "s3_unseen_language"
    slice_of[~is_lang & is_dom] = "s4_unseen_domain"
    slice_of[is_fam & ~is_lang & ~is_dom] = "s2_unseen_generator"

    # Held-out human code carries no generator family, so it must be shared
    # into the generator slice explicitly or s2 would contain only positives.
    human_pool = df.index[trainable & (df["label"] == 0)]
    n_human_s2 = int(min(len(human_pool), max((slice_of == "s2_unseen_generator").sum(), 1)))
    if n_human_s2:
        picked = rng.choice(human_pool, size=n_human_s2, replace=False)
        slice_of.loc[picked] = "s2_unseen_generator"
        trainable.loc[picked] = False

    # Remaining trainable rows split into train / val / s1, grouped by problem.
    pool = df.index[trainable]
    groups = np.asarray(df.loc[pool, "problem_id"].unique(), dtype=object)
    rng.shuffle(groups)
    n = len(groups)
    n_test = int(n * 0.12)
    n_val = int(n * cfg.splits.val_fraction)
    g_test = set(groups[:n_test])
    g_val = set(groups[n_test : n_test + n_val])

    split = pd.Series("unused", index=df.index, dtype=object)
    pid = df["problem_id"]
    split.loc[pool] = np.where(
        pid.loc[pool].isin(g_test), "s1_in_distribution",
        np.where(pid.loc[pool].isin(g_val), "val", "train"),
    )
    slice_of.loc[pool] = split.loc[pool]

    # OOD slices must not share a problem_id with train, or the shift is fake.
    train_pids = set(df.loc[split == "train", "problem_id"])
    for s in ("s2_unseen_generator", "s3_unseen_language", "s4_unseen_domain", "s5_compound"):
        m = slice_of == s
        leak = m & pid.isin(train_pids)
        if leak.any():
            slice_of[leak] = "unused"
            print(f"  [dedup] dropped {int(leak.sum()):,} rows from {s} sharing a problem_id with train")

    df["slice"] = slice_of
    df["split"] = np.where(
        slice_of.isin(["train", "val"]), slice_of,
        np.where(slice_of == "unused", "unused", "test"),
    )
    return df


SLICES = [
    "s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
    "s4_unseen_domain", "s5_compound",
]


def report(df: pd.DataFrame, cfg) -> dict:
    out: dict = {}
    print(f"\n{'slice':22s} {'rows':>8s} {'human':>7s} {'mach':>7s} {'hybrid':>7s} {'adv':>7s}")
    for s in ["train", "val"] + SLICES:
        g = df[df["slice"] == s]
        counts = [int((g["label"] == i).sum()) for i in range(4)]
        out[s] = {"rows": len(g), "labels": counts}
        print(f"{s:22s} {len(g):>8,} {counts[0]:>7,} {counts[1]:>7,} {counts[2]:>7,} {counts[3]:>7,}")
    unused = int((df["slice"] == "unused").sum())
    print(f"{'unused':22s} {unused:>8,}")
    out["unused"] = {"rows": unused}

    print("\nslice guarantees")
    train_pids = set(df.loc[df["slice"] == "train", "problem_id"])
    for s in SLICES:
        g = df[df["slice"] == s]
        if not len(g):
            print(f"  {s:22s} EMPTY")
            continue
        overlap = len(set(g["problem_id"]) & train_pids)
        n_lab = int(g["label"].nunique())
        flag = "ok " if overlap == 0 else "LEAK"
        small = "" if len(g) >= cfg.splits.min_slice_size else "  [warn] under min_slice_size"
        print(f"  {s:22s} problem_id overlap={overlap:<4d} labels={n_lab} {flag}{small}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    args = ap.parse_args()
    cfg = C.load(args.config)
    d = C.ROOT / cfg.data.cache_dir

    df = pd.read_parquet(d / "filtered.parquet")
    print(f"[splits] assigning over {len(df):,} rows")
    print(f"  holdout families : {cfg.splits.holdout_families}")
    print(f"  holdout languages: {cfg.splits.holdout_languages}")
    print(f"  holdout sources  : {cfg.splits.holdout_sources}")
    df = assign(df, cfg)
    stats = report(df, cfg)

    df.to_parquet(d / "splits.parquet", index=False)
    with open(C.reports(cfg) / "splits.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[done] splits.parquet ({len(df):,} rows)")


if __name__ == "__main__":
    main()
