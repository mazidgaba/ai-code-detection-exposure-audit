"""How much of the collapse is training noise?

Bootstrap intervals elsewhere in this work resample the evaluation set while
the weights stay fixed. They say nothing about how far the result moves when
the same recipe is trained again from a different initialisation, and a
reviewer is right to treat that as an open question rather than an answered
one.

This reads every per-seed report it can find and reports the spread. The
comparison that matters is not the spread itself but its size against the
effect: if the S1-to-S5 drop is an order of magnitude larger than the
seed-to-seed standard deviation, the collapse is not an artefact of one
unlucky initialisation.

Reports are discovered rather than listed, so adding a seed means dropping its
JSON in and re-running.

    python -m aicd.eval.seed_variance
"""
from __future__ import annotations

import glob
import io
import json
import os

import numpy as np

from aicd import config as C

ORDER = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
         "s4_unseen_domain", "s5_compound"]

# The run reported in the manuscript. Its seed is the project default, and its
# report lives under the kaggle subdirectory rather than beside the seed runs.
CANONICAL = ("kaggle/branch_a_base.json", "paper (seed 20260818)")


def find_reports():
    """Locate the canonical run and every branch_a_seed*.json beside it."""
    rep = C.ROOT / "eval" / "reports"
    out = []

    p = rep / CANONICAL[0]
    if p.exists():
        out.append((CANONICAL[1], json.load(io.open(p, encoding="utf-8"))))

    seen = set()
    roots = [str(rep), os.path.join(str(C.ROOT.parent), "kaggle_runs", "results")]
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "**", "branch_a_seed*.json"),
                                  recursive=True)):
            name = os.path.basename(f)[len("branch_a_"):-len(".json")]
            if name in seen:
                continue
            seen.add(name)
            out.append((name, json.load(io.open(f, encoding="utf-8"))))
    return out


def main() -> None:
    runs = find_reports()
    if len(runs) < 2:
        raise SystemExit(
            f"found {len(runs)} run(s); at least two are needed for a spread.\n"
            "Drop each seed's branch_a_seed<N>.json under kaggle_runs/results/.")

    print(f"{'run':26s} " + " ".join(f"{s.split('_')[0].upper():>8s}" for s in ORDER))
    print("-" * (26 + 9 * len(ORDER)))
    table = {}
    for name, d in runs:
        sl = d.get("slices", {})
        row = [sl[s]["macro_f1"] if s in sl else np.nan for s in ORDER]
        table[name] = row
        print(f"{name:26s} " + " ".join(f"{v:8.4f}" for v in row))

    arr = np.array(list(table.values()), dtype=float)
    mean, sd = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0, ddof=1)
    print("-" * (26 + 9 * len(ORDER)))
    print(f"{'mean':26s} " + " ".join(f"{v:8.4f}" for v in mean))
    print(f"{'sd':26s} " + " ".join(f"{v:8.4f}" for v in sd))

    # The headline: the effect against the noise.
    drop = mean[0] - mean[-1]
    worst_sd = float(np.nanmax(sd))
    print()
    print(f"n runs                        : {len(runs)}")
    print(f"mean S1 -> S5 drop            : {drop:.4f}")
    print(f"largest per-condition sd      : {worst_sd:.4f}")
    print(f"drop / sd                     : {drop / max(worst_sd, 1e-9):.0f}x")
    print()
    if drop > 10 * worst_sd:
        print("The collapse is far larger than the spread across seeds, so it")
        print("is not an artefact of one initialisation.")
    else:
        print("The spread is not small against the effect. Report both, and do")
        print("not lean on a single run.")

    out = {
        "n_runs": len(runs),
        "runs": {k: dict(zip(ORDER, v)) for k, v in table.items()},
        "mean": dict(zip(ORDER, mean.tolist())),
        "sd": dict(zip(ORDER, sd.tolist())),
        "mean_s1_to_s5_drop": float(drop),
        "largest_sd": worst_sd,
        "drop_over_sd": float(drop / max(worst_sd, 1e-9)),
    }
    dest = C.ROOT / "eval" / "reports" / "seed_variance.json"
    os.makedirs(dest.parent, exist_ok=True)
    io.open(dest, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
