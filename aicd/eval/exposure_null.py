"""Is 35.6% a lot? The exposure audit against a null it has never been given.

The audit reports that 35.6% of the training split a published detector names
carries at least one of the seven categories our benchmark withholds. The paper
uses that figure to argue those categories were not novel to the detector.

A reader can reasonably ask whether 35.6% is remarkable. If seven categories
drawn at random from this collection typically cover a third of it, then the
audit has measured the size of seven categories rather than anything about the
benchmark's choices, and the finding is arithmetic.

This builds the null. It draws seven categories at random under the same
structure the real withholding uses, two generator families, two languages and
three sources, and measures the share of training rows they touch. The observed
value is then placed in that distribution.

Two outcomes, and both are worth having:

  If 35.6% sits in the body of the null, the honest reading is that any seven
  categories of this shape would cover a comparable share, and the audit's force
  comes from the fact that they are present at all rather than from how many
  rows they touch.

  If it sits in the tail, the categories a benchmark chooses to withhold are
  unusually well represented in the training data, which is a stronger claim
  than the paper currently makes.

    python -m aicd.eval.exposure_null
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aicd import config as C

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "aicd" / "eval" / "reports"


def share(df: pd.DataFrame, fams, langs, srcs) -> float:
    m = (df["model_family"].isin(fams) | df["language"].isin(langs)
         | df["source"].isin(srcs))
    return float(m.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    cfg = C.load(args.config)
    # The population is DroidCollection's own training shards, not our split.
    # Ours contains no withheld category by construction, so measuring the null
    # against it returns zero and answers a question nobody asked. The audit
    # reads the shards; so must its null.
    from aicd.eval.exposure_audit import (COLUMN, WITHHELD,
                                          load_train_categories)
    raw = load_train_categories()
    # Normalised exactly as the audit normalises, so the observed value this
    # null is compared against is the audit's own number and not a near miss.
    norm = {axis: raw[COLUMN[axis]].astype(str).str.strip().str.lower()
            for axis in WITHHELD}

    def union_share(picks: dict) -> float:
        m = None
        for axis, cats in picks.items():
            hit = norm[axis].isin(cats)
            m = hit if m is None else (m | hit)
        return float(m.mean())

    real = union_share({a: set(c) for a, c in WITHHELD.items()})
    axes = list(WITHHELD)
    counts = {a: len(WITHHELD[a]) for a in axes}
    n_f, n_l, n_s = (counts[axes[0]], counts[axes[1]], counts[axes[2]])

    # The null draws from the categories the collection actually contains,
    # including those that were withheld, because a benchmark designer choosing
    # what to hold out chooses from the same menu.
    pools = {a: sorted(set(norm[a].dropna())) for a in axes}
    print("collection offers " + ", ".join(
        f"{len(pools[a])} {a}" for a in axes))
    print("the benchmark withholds " + ", ".join(
        f"{counts[a]} {a}" for a in axes))
    print(f"observed share of the training split touched: {real:.4f}")

    rng = np.random.default_rng(args.seed)
    null = np.empty(args.draws)
    for i in range(args.draws):
        picks = {a: set(rng.choice(pools[a],
                                   size=min(counts[a], len(pools[a])),
                                   replace=False))
                 for a in axes}
        null[i] = union_share(picks)

    pct = float((null <= real).mean())
    out = {"observed": real, "draws": args.draws,
           "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
           "null_median": float(np.median(null)),
           "null_p05": float(np.percentile(null, 5)),
           "null_p95": float(np.percentile(null, 95)),
           "percentile_of_observed": pct,
           "withheld_counts": counts,
           "pool_sizes": {a: len(pools[a]) for a in axes}}

    print(f"\nnull over {args.draws:,} random draws of the same shape:")
    print(f"  mean   {out['null_mean']:.4f}   sd {out['null_sd']:.4f}")
    print(f"  median {out['null_median']:.4f}")
    print(f"  90% of draws fall in [{out['null_p05']:.4f}, {out['null_p95']:.4f}]")
    print(f"\nobserved 35.6% sits at the {pct:.1%} percentile of that null.")
    print()
    if 0.05 <= pct <= 0.95:
        out["reading"] = "typical"
        print("Typical. Seven categories of this shape usually touch a comparable")
        print("share of the split, so the audit's force is that the categories are")
        print("present at all, not that they cover a third of the rows. The paper")
        print("should make the qualitative claim and use 35.6% as illustration.")
    elif pct > 0.95:
        out["reading"] = "unusually high"
        print("Unusually high. The categories this benchmark withholds are better")
        print("represented in the training split than a random choice would be,")
        print("which is a stronger claim than the paper currently makes.")
    else:
        out["reading"] = "unusually low"
        print("Unusually low, which would weaken the argument rather than support")
        print("it. Report it plainly.")

    dest = REPORTS / "exposure_null.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
