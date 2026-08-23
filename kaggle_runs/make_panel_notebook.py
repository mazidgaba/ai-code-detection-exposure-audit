"""Generate one notebook per detector in the E3 panel.

Reviewer U3, raised independently by all three reviewers: the paper writes
"detectors key on surface lexical habit" while having evaluated exactly one
detector, and one of those was a reconstruction. Every plural claim in the
paper rests on a panel that does not exist yet.

Two of the four members need no new training code at all. TLModel takes an
encoder, reads `last_hidden_state` and the encoder's own `hidden_size`, and
mean-pools; nothing in it is specific to ModernBERT. So CodeBERT and UniXcoder
drop in through a config file, trained on the identical split with the
identical recipe, which is what makes them comparable rather than merely
additional.

    python kaggle_runs/make_panel_notebook.py --detectors codebert,unixcoder
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

DETECTORS = {
    "codebert": {
        "config": "kaggle_panel_codebert",
        "tag": "panel_codebert",
        "name": "CodeBERT",
        "model": "microsoft/codebert-base",
        "hours": "8-9 h",
        "note": """CodeBERT is RoBERTa-shaped at hidden size 768 with 512 maximum positions,
which is exactly the sequence length already in use, so nothing about the
recipe has to change to accommodate it.""",
    },
    "unixcoder": {
        "config": "kaggle_panel_unixcoder",
        "tag": "panel_unixcoder",
        "name": "UniXcoder",
        "model": "microsoft/unixcoder-base",
        "hours": "8-9 h",
        "note": """UniXcoder is already cited in the paper's Related Work and has never been
evaluated in it. Same shape as CodeBERT, so the same config-only treatment
applies.""",
    },
}

INTRO = """# E3 panel: {name}

{note}

## Why this exists

The paper currently says "the detectors" while having evaluated one. Reviewer
U3 was raised independently by all three reviewers, and it is the kind of
finding that costs nothing to fix and a great deal to leave. After this run and
its sibling, the honest phrasing becomes "the detectors we evaluate", and the
mechanism claims have more than one system standing behind them.

What makes this a comparison rather than an addition is that {name} is trained
on the **identical split** with the **identical recipe** and the **identical
seed**. Only `{config}.yaml` differs from `kaggle.yaml`, and only in
`base_model`. The training code is untouched: `TLModel` reads
`last_hidden_state` and the encoder's own `hidden_size`, so any HuggingFace
encoder substitutes cleanly.

Model: `{model}`

## Settings

| Setting | Value |
|---|---|
| **Accelerator** | `GPU T4 x2` |
| **Internet** | `On` |
| **Persistence** | `Files only` |

**Save Version -> Save & Run All (Commit).** Budget {hours}. The trainer is
given `--max-hours 10`, so it stops after the last epoch that fits rather than
being killed mid-epoch. Re-run with the previous version's output attached to
continue from the checkpoint.
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
print(f"{{'condition':24s}} {{'{name}':>12s}} {{'ModernBERT':>12s}} {{'delta':>8s}}")
print("-" * 60)
for s, ref in REFERENCE.items():
    if s in r:
        v = r[s]["macro_f1"]
        print(f"{{s:24s}} {{v:12.4f}} {{ref:12.4f}} {{v-ref:+8.4f}}")
s1 = r.get("s1_in_distribution", {{}}).get("macro_f1")
s5 = r.get("s5_compound", {{}}).get("macro_f1")
if s1 and s5:
    print(f"\\nS1 -> S5: {{s1:.4f}} -> {{s5:.4f}}  (drop {{s1-s5:.4f}})")
    print(f"ModernBERT drop:                   {{0.8977-0.2378:.4f}}")
    print()
    if s5 < 0.45:
        print("{name} collapses under compound shift as well. The plural claim")
        print("in the paper now has a second, independently trained system")
        print("standing behind it.")
    else:
        print("{name} does NOT collapse. That is a substantial finding and it")
        print("narrows the paper's claim to the architecture, not to detectors")
        print("in general. Report it plainly rather than burying it.")'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", default="codebert,unixcoder")
    args = ap.parse_args()

    for name in [d.strip() for d in args.detectors.split(",") if d.strip()]:
        v = DETECTORS[name]
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
            G.md(f"## 4. Train {v['name']}"), G.code(TRAIN.format(**v)),
            G.md("## 5. Against ModernBERT on the same split"),
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
        p = HERE / f"9_e3_{name}.ipynb"
        p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"wrote {p.name}  ({len(cells)} cells, all code parses)")


if __name__ == "__main__":
    main()
