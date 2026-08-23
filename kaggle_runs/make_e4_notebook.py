"""Generate the E4 seed notebook, reusing the E1 generator's building blocks.

All three reviewers raised the same objection: three seeds is not enough. With
three runs there are two degrees of freedom, so the 95% upper bound on the true
standard deviation is about six times the sample one, which means the paper's
"48 times the training noise" could honestly be eight times. Two more seeds
takes the sweep to five and lets the claim be stated as a paired difference
with an interval instead of a ratio.

    python kaggle_runs/make_e4_notebook.py
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_e1_notebooks as G  # noqa: E402

TRAIN = '''run(["-m", "aicd.models.modernbert_triplet",
     "--config", "kaggle_seed{seed}.yaml",
     "--tag", "seed{seed}", "--max-hours", "10", "--resume"])
elapsed("seed {seed} done")'''

REPORT = r'''import numpy as np
rep = WORK / "aicd" / "eval" / "reports"

runs = {"seed 20260818 (paper)": {"s1_in_distribution": 0.8977,
                                  "s5_compound": 0.2378}}
for seed in [1, 2, 3, 4]:
    f = rep / f"branch_a_seed{seed}.json"
    if f.exists():
        r = json.load(open(f))["slices"]
        runs[f"seed {seed}"] = {k: r[k]["macro_f1"]
                                for k in ["s1_in_distribution", "s5_compound"]
                                if k in r}

print(f"{'run':26s} {'S1':>9s} {'S5':>9s}")
print("-" * 47)
for name, v in runs.items():
    print(f"{name:26s} {v.get('s1_in_distribution', float('nan')):9.4f} "
          f"{v.get('s5_compound', float('nan')):9.4f}")

s1 = [v["s1_in_distribution"] for v in runs.values() if "s1_in_distribution" in v]
s5 = [v["s5_compound"] for v in runs.values() if "s5_compound" in v]
if len(s1) > 2:
    drops = [a - b for a, b in zip(s1, s5)]
    print()
    print(f"n = {len(s1)} seeds")
    print(f"S1    mean {np.mean(s1):.4f}  sd {np.std(s1, ddof=1):.4f}")
    print(f"S5    mean {np.mean(s5):.4f}  sd {np.std(s5, ddof=1):.4f}")
    print(f"drop  mean {np.mean(drops):.4f}  sd {np.std(drops, ddof=1):.4f}")
    print()
    print("Report the PAIRED drop with its interval, not the ratio of a drop")
    print("to a standard deviation. That is reviewer finding 4.3: with three")
    print("seeds the 48x claim was not supportable, and with five the honest")
    print("statement is a mean and an interval that excludes zero.")'''

INTRO = """# E4: seed {seed}

All three reviewers said three seeds is not enough, finding U4. With three runs
there are two degrees of freedom, so the 95% upper bound on the true standard
deviation is roughly six times the sample one: the paper's "48 times the
training noise" could honestly be eight times. This takes the sweep to five.

The corpus is built once from `kaggle.yaml` and every seed trains on that same
`splits.parquet`, so what this measures is training variance and not split
variance. That distinction matters, because `splits.py` seeds its own RNG from
`project.seed`, and rebuilding the corpus per seed would have quietly turned a
seed sweep into a sweep over partitions as well.

## Settings

| Setting | Value |
|---|---|
| **Accelerator** | `GPU T4 x2` |
| **Internet** | `On` |
| **Persistence** | `Files only` |

**Save Version -> Save & Run All (Commit).** Budget about 10 h, which is also
the `--max-hours` the trainer is given, so it stops after the last epoch that
fits rather than being killed mid-epoch. Re-run this notebook with the previous
version's output attached to continue where it stopped.

One seed per notebook, deliberately. A single notebook training both seeds runs
them back to back and takes twice the wall-clock for no benefit, whereas two
notebooks on two accounts finish together.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="3,4",
                    help="one notebook per seed, so they can run on separate "
                         "accounts at the same time instead of queueing")
    args = ap.parse_args()

    for seed in [int(x) for x in args.seeds.split(",")]:
        cells = [
            G.md(INTRO.format(seed=seed)),
            G.code(G.SETUP),
            G.md("## 1. Code"), G.code(G.CODE),
            G.md("## 2. Dependencies"), G.code(G.DEPS),
            G.md("## 3. Resume, if a previous run is attached"), G.code(G.RESTORE.format(tag=f"seed{seed}")),
            G.md("## 3. Build the corpus\n\nOne train shard, exactly the "
                 "original GPU build: 493,850 raw rows filtering to 417,645, "
                 "of which 196,854 are training."),
            G.code(G.BUILD),
            G.md(f"## 4. Train seed {seed}"), G.code(TRAIN.format(seed=seed)),
            G.md("## 5. Spread across every seed present"), G.code(REPORT),
            G.md("## 6. Save"), G.code(G.SAVE),
        ]
        for c in cells:
            if c["cell_type"] == "code":
                ast.parse("".join(c["source"]))
        nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
              "metadata": {"kernelspec": {"display_name": "Python 3",
                                          "language": "python",
                                          "name": "python3"},
                           "language_info": {"name": "python"}}}
        p = HERE / f"7_e4_seed{seed}.ipynb"
        p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"wrote {p.name}  ({len(cells)} cells, all code parses)")


if __name__ == "__main__":
    main()
