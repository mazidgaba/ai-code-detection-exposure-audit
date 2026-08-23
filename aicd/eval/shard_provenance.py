"""Where every adversarial row in the corpus came from.

A reviewer found what looked like a contradiction. The paper says all 129,115
adversarial rows sit in training shard 2, and separately that the GPU build
reads one training shard plus dev and test. Together those imply the build has
no adversarial data at all, yet its training split alone has 6,123.

The resolution is that the first statement is incomplete rather than false. It
is true of the three training shards: shard 2 holds all 129,115 of theirs, and
shards 0 and 1 hold none. But dev and test carry another 32,279 between them,
which the sentence never mentions, and those are exactly what the build draws
on. Written as it stands, the sentence reads as a claim about the whole
collection.

This module recomputes the provenance from the raw parquet files, so the
paragraph that replaces it in the paper rests on a measurement rather than on
a recollection of how the loader behaves.

    python -m aicd.eval.shard_provenance
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from aicd.data.normalize import map_label

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "aicd" / "artifacts" / "data" / "hf"
LABELS = ROOT / "aicd" / "artifacts" / "kaggle" / "labels.parquet"
LABEL_NAMES = ["human", "machine", "hybrid", "adversarial"]
ADVERSARIAL = 3


def raw_counts() -> dict:
    out = {}
    for f in sorted(glob.glob(str(RAW / "*.parquet"))):
        t = pq.read_table(f, columns=["Label"])
        c = collections.Counter(map_label(v) for v in t.column("Label").to_pylist())
        out[os.path.basename(f)] = {"rows": int(sum(c.values())),
                                    "adversarial": int(c.get(ADVERSARIAL, 0))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = raw_counts()
    print(f"{'raw file':34s} {'rows':>9s} {'adversarial':>12s} {'share':>7s}")
    print("-" * 66)
    for name, v in raw.items():
        share = v["adversarial"] / max(v["rows"], 1)
        print(f"  {name:32s} {v['rows']:>9,} {v['adversarial']:>12,} {share:>6.1%}")

    train_adv = sum(v["adversarial"] for k, v in raw.items() if k.startswith("train"))
    eval_adv = sum(v["adversarial"] for k, v in raw.items()
                   if k.startswith(("dev", "test")))
    print(f"\n  across the three training shards : {train_adv:,}")
    print(f"  across dev and test              : {eval_adv:,}")
    print("  The paper's sentence is true of the first number and silent about"
          "\n  the second, which is the one the build actually uses.")

    report = {"raw": raw, "adversarial_in_training_shards": train_adv,
              "adversarial_in_dev_and_test": eval_adv}

    if LABELS.exists():
        lab = pd.read_parquet(LABELS, columns=["label", "slice", "orig_split"])
        adv = lab[lab["label"] == ADVERSARIAL]
        by_src = adv["orig_split"].value_counts().to_dict()
        by_slice = adv["slice"].value_counts().to_dict()
        report["build"] = {
            "adversarial_rows": int(len(adv)),
            "by_original_split": {k: int(v) for k, v in by_src.items()},
            "by_condition": {k: int(v) for k, v in by_slice.items()},
            "survival_rate": float(len(adv) / max(eval_adv, 1)),
        }
        print(f"\n  in the built corpus              : {len(adv):,} "
              f"({len(adv) / max(eval_adv, 1):.1%} of what dev and test offered)")
        print("  sourced from:", ", ".join(f"{k} {v:,}" for k, v in by_src.items()))
        print("  landing in  :", ", ".join(f"{k} {v:,}"
                                           for k, v in sorted(by_slice.items())))
        if by_src and "train" in by_src:
            print("\n  NOTE: some adversarial rows came from the raw training "
                  "shards after all, which contradicts the reading above.")

    dest = Path(args.out) if args.out else (
        ROOT / "aicd" / "eval" / "reports" / "shard_provenance.json")
    os.makedirs(dest.parent, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten -> {dest}")


if __name__ == "__main__":
    main()
