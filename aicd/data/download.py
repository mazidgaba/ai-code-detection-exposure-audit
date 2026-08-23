"""Phase 01a: pull DroidCollection and cache as parquet.

DroidCollection schema (verified against the live dataset):
    Code, Generator, Generation_Mode, Source, Language,
    Sampling_Params, Rewriting_Params, Label, Model_Family

Files are fetched straight from the HF resolve endpoint rather than through
`datasets`, so this works with just pandas + pyarrow and resumes cleanly.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

from aicd import config as C

REPO = "project-droid/DroidCollection"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/data"

FILES = {
    "dev": ["dev-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
    "train": [
        "train-00000-of-00003.parquet",
        "train-00001-of-00003.parquet",
        "train-00002-of-00003.parquet",
    ],
}


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [have] {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    tmp = dest.with_suffix(".part")
    print(f"  [get ] {dest.name} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "aicd/1.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r        {done / 1e6:7.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.rename(dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--train-shards", type=int, default=1,
                    help="how many of the 3 train shards to pull")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = C.load(args.config)
    out = C.ROOT / cfg.data.cache_dir
    raw = out / "hf"
    raw.mkdir(parents=True, exist_ok=True)
    max_rows = cfg.data.max_rows_per_split

    for split, names in FILES.items():
        dest = out / f"raw_{split}.parquet"
        if dest.exists() and not args.force:
            print(f"[skip] {dest.name} exists")
            continue

        use = names[: args.train_shards] if split == "train" else names
        frames = []
        print(f"[fetch] {split} ({len(use)} file(s), max_rows={max_rows})")
        # Read every requested shard before sampling. The classes are not
        # spread evenly across shards -- all adversarial samples live in
        # train shard 2 -- so stopping early silently drops a whole class.
        for n in use:
            p = download(f"{BASE}/{n}", raw / n)
            frames.append(pd.read_parquet(p))

        df = pd.concat(frames, ignore_index=True)
        if max_rows and len(df) > max_rows:
            # Stratified by label so a cap never starves a class. Index-based
            # rather than groupby.apply, which drops the grouping column on
            # pandas 3.
            keep = []
            for _, idx in df.groupby("Label").groups.items():
                n = max(1, int(round(max_rows * len(idx) / len(df))))
                keep.append(pd.Index(idx).to_series().sample(
                    n=min(n, len(idx)), random_state=cfg.project.seed))
            df = df.loc[pd.concat(keep).values].reset_index(drop=True)
        df["orig_split"] = split
        df.to_parquet(dest, index=False)
        print(f"[done] {dest.name}: {len(df):,} rows, {dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
