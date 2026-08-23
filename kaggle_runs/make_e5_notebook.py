"""Generate one E5 ablation notebook per configuration.

Reviewer S3 found a direct contradiction: the contributions list claimed a
control excluding "model capacity and free-parameter choice", and no such
control had been run. The claim has since been removed from the paper. These
runs are what would let a true version of it go back in.

Every axis was already a configuration knob, so this needs no new training
code, only configs that differ in exactly one thing each:

    kaggle_e5_large       ModernBERT-large instead of base   (capacity)
    kaggle_e5_triplet0    triplet term removed
    kaggle_e5_triplet02   triplet weight doubled
    kaggle_e5_proj256     projection 128 -> 256

One notebook per configuration, so they can run on separate accounts at the
same time rather than queueing behind each other.

    python kaggle_runs/make_e5_notebook.py --configs large,triplet0
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

VARIANTS = {
    "large": {
        "config": "kaggle_e5_large",
        "tag": "e5_large",
        "axis": "encoder capacity",
        "hours": "about 26 h over three sessions",
        "note": """The first attempt at this run died with CUDA out of memory, having taken all
but 57 MiB of a 14.56 GiB T4. ModernBERT-large is 395M parameters against
base's 149M, and at sequence length 512 its activations do not fit.

The fix is gradient checkpointing, which recomputes activations during the
backward pass instead of storing them. It is the right lever precisely because
it changes nothing about the result: the gradients and the parameter updates
are what they would have been without it, and only the wall clock moves, by
roughly a third. Reducing the batch further or shortening the sequence would
have been cheaper and would have changed the experiment.

Two deviations from the reference configuration remain, and both belong in the
paper: batch size 16 rather than 32, and checkpointing on. Sequence length
stays at 512 on purpose, because shortening it would mean the large model saw
less of each file than the reference did, and the ablation could then no longer
separate capacity from truncation.

**This needs three sessions.** Each stops cleanly after the epoch that fits in
`--max-hours 10`; re-run with the previous version's output attached.""",
    },
    "triplet0": {
        "config": "kaggle_e5_triplet0",
        "tag": "e5_triplet0",
        "axis": "triplet term removed",
        "hours": "9-10 h",
        "note": """Pure class-weighted cross-entropy, with the batch-hard triplet term switched
off entirely. If the collapse is present here too, it is not a property of the
metric-learning objective.""",
    },
    "augment": {
        "config": "kaggle_e8_augment",
        "tag": "e8_augment",
        "axis": "training through the rewrites (E8 mitigation)",
        "hours": "8-9 h",
        "note": """Half the training rows are rewritten before training: identifiers renamed to
opaque but internally consistent names, and whitespace normalised. Labels are
untouched, because a renamed machine-generated file is still machine-generated.

Both outcomes are useful. If the collapse softens, the paper gains a mitigation
and stops being purely diagnostic. If it does not, the finding is quotable and
stronger than what the paper currently claims: augmenting away the surface cue
does not restore transfer, so the dependence is deeper than formatting.

The rewrite is applied once before training rather than afresh each epoch, so
the corpus is reproducible from the seed and a resumed session continues on
exactly the data it started with.""",
    },
    "triplet02": {
        "config": "kaggle_e5_triplet02",
        "tag": "e5_triplet02",
        "axis": "triplet weight doubled",
        "hours": "9-10 h",
        "note": "Triplet weight 0.2 against the default 0.1.",
    },
    "proj256": {
        "config": "kaggle_e5_proj256",
        "tag": "e5_proj256",
        "axis": "projection dimension doubled",
        "hours": "9-10 h",
        "note": "Projection 256 against the default 128.",
    },
}

INTRO = """# E5: {axis}

{note}

## What this is for

The reporting rule matters and is easy to get wrong. The finding is **not** that
this setting is better or optimal. It is that the collapse under compound shift
is present in every configuration we tried, so it cannot be dismissed as an
artefact of one particular set of free parameters.

The corpus is built exactly as the original GPU build, and only
`{config}.yaml` differs from `kaggle.yaml`, in one axis.

## Settings

| Setting | Value |
|---|---|
| **Accelerator** | `GPU T4 x2` |
| **Internet** | `On` |
| **Persistence** | `Files only` |

**Save Version -> Save & Run All (Commit).** Budget {hours}. The trainer is
given `--max-hours 10`, so it stops after the last epoch that fits rather than
being killed mid-epoch, and re-running this notebook with the previous
version's output attached continues from that checkpoint.
"""

TRAIN = '''run(["-m", "aicd.models.modernbert_triplet",
     "--config", "{config}.yaml",
     "--tag", "{tag}", "--max-hours", "10", "--resume"])
elapsed("{tag} trained and evaluated")'''

REPORT = '''rep = WORK / "aicd" / "eval" / "reports" / "branch_a_{tag}.json"
r = json.load(open(rep))["slices"]
REFERENCE = {{"s1_in_distribution": 0.8977, "s2_unseen_generator": 0.8685,
             "s3_unseen_language": 0.5667, "s4_unseen_domain": 0.4029,
             "s5_compound": 0.2378}}
print(f"{{'condition':24s}} {{'this run':>10s}} {{'reference':>10s}} {{'delta':>8s}}")
print("-" * 56)
for s, ref in REFERENCE.items():
    if s in r:
        v = r[s]["macro_f1"]
        print(f"{{s:24s}} {{v:10.4f}} {{ref:10.4f}} {{v-ref:+8.4f}}")
s1 = r.get("s1_in_distribution", {{}}).get("macro_f1")
s5 = r.get("s5_compound", {{}}).get("macro_f1")
if s1 and s5:
    print(f"\\nS1 -> S5: {{s1:.4f}} -> {{s5:.4f}}  (drop {{s1-s5:.4f}})")
    print(f"reference drop:                    {{0.8977-0.2378:.4f}}")
    if s5 < 0.45:
        print("\\nThe collapse is present in this configuration too.")
    else:
        print("\\nThe collapse is NOT present here. That is a real finding and")
        print("changes the paper: part of the effect was configuration-specific.")'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="large,triplet0,triplet02,proj256")
    args = ap.parse_args()

    for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
        v = VARIANTS[name]
        cells = [
            G.md(INTRO.format(**v)),
            G.code(G.SETUP),
            G.md("## 1. Code"), G.code(G.CODE),
            G.md("## 2. Dependencies"), G.code(G.DEPS),
            G.md("## 3. Resume, if a previous run is attached"), G.code(G.RESTORE.format(tag=v["tag"])),
            G.md("## 3. Build the corpus\n\nOne train shard, exactly the "
                 "original GPU build: 493,850 raw rows filtering to 417,645, "
                 "of which 196,854 are training."),
            G.code(G.BUILD),
            G.md(f"## 4. Train with {v['axis']}"),
            G.code(TRAIN.format(**v)),
            G.md("## 5. Against the reference configuration"),
            G.code(REPORT.format(**v)),
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
        p = HERE / f"8_e5_{name}.ipynb"
        p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"wrote {p.name}  ({len(cells)} cells, all code parses)")


if __name__ == "__main__":
    main()
