"""Phase 01c: quality filtering, mirroring the published Droid / CoDet-M4 pipeline.

Steps, each logged with the row count it removes:
  1. drop empty / trivially short samples
  2. drop samples that fail to parse into an AST (tree-sitter)
  3. drop samples outside the 5th-95th token-count percentile, PER LANGUAGE
  4. MinHash-deduplicate at Jaccard 0.85
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

from aicd import config as C
from aicd.features.ast_feats import get_parser, parse_ok

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def n_tokens(code: str) -> int:
    return len(TOKEN_RE.findall(code))


def step_min_chars(df: pd.DataFrame, min_chars: int) -> pd.DataFrame:
    return df[df["code"].str.len() >= min_chars]


def step_parseable(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    cache: dict[str, object] = {}
    total = len(df)
    for i, (lang, code) in enumerate(zip(df["language"], df["code"])):
        if lang not in cache:
            cache[lang] = get_parser(lang)
        keep.append(parse_ok(cache[lang], code))
        if i % 20000 == 0:
            print(f"    parse {i:,}/{total:,}", flush=True)
    return df[pd.Series(keep, index=df.index)]


def step_length_percentile(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    df = df.copy()
    df["_ntok"] = df["code"].map(n_tokens)
    keep = pd.Series(False, index=df.index)
    for lang, grp in df.groupby("language"):
        a, b = grp["_ntok"].quantile([lo / 100, hi / 100])
        keep.loc[grp.index] = grp["_ntok"].between(a, b)
    return df[keep].drop(columns=["_ntok"])


NUM_PERM = 32
MAX_SHINGLES = 160


def _shingles(code: str, k: int = 5, cap: int = MAX_SHINGLES) -> list[bytes]:
    """Up to `cap` k-char shingles, strided evenly across the file.

    Capping matters: a 5k-char file yields ~1600 shingles at stride 3, and
    hashing all of them for every row makes dedup the slowest step in the
    pipeline by an order of magnitude. An evenly-spread sample of 160 is
    plenty to catch near-duplicates at Jaccard 0.85.
    """
    n = max(len(code) - k, 1)
    stride = max(1, n // cap)
    return [code[i: i + k].encode("utf-8", "ignore") for i in range(0, n, stride)][:cap]


def step_minhash_dedup(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("  [warn] datasketch missing, falling back to exact-hash dedup")
        return df.drop_duplicates(subset=["code"])

    # Exact duplicates first -- cheap, and it shrinks the LSH workload.
    before = len(df)
    df = df.drop_duplicates(subset=["code"])
    print(f"    exact-dup pass removed {before - len(df):,}")

    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    keep = []
    total = len(df)
    for i, (idx, code) in enumerate(zip(df.index, df["code"])):
        m = MinHash(num_perm=NUM_PERM)
        m.update_batch(_shingles(code))
        if lsh.query(m):
            continue
        lsh.insert(str(idx), m)
        keep.append(idx)
        if i % 20000 == 0:
            print(f"    minhash {i:,}/{total:,}", flush=True)
    return df.loc[keep]


def run(df: pd.DataFrame, cfg) -> pd.DataFrame:
    log = []
    steps = [
        ("min_chars", lambda d: step_min_chars(d, cfg.data.min_chars)),
        ("parseable", step_parseable),
        ("length_pct", lambda d: step_length_percentile(d, cfg.data.token_pct_low, cfg.data.token_pct_high)),
        ("minhash_dedup", lambda d: step_minhash_dedup(d, cfg.data.minhash_threshold)),
    ]
    for name, fn in steps:
        before = len(df)
        df = fn(df)
        log.append((name, before, len(df), before - len(df)))
        print(f"  {name:16s} {before:>8,} -> {len(df):>8,}  (-{before-len(df):,})")

    print("\nfilter log")
    print(f"{'step':16s} {'before':>9s} {'after':>9s} {'removed':>9s} {'pct':>7s}")
    for name, b, a, r in log:
        print(f"{name:16s} {b:>9,} {a:>9,} {r:>9,} {100*r/max(b,1):>6.1f}%")
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    args = ap.parse_args()
    cfg = C.load(args.config)
    d = C.ROOT / cfg.data.cache_dir

    df = pd.read_parquet(d / "normalized.parquet")
    print(f"[filter] starting from {len(df):,} rows\n")
    df = run(df, cfg)
    dest = d / "filtered.parquet"
    df.to_parquet(dest, index=False)
    print(f"\n[done] {dest.name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
