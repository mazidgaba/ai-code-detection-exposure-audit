"""Generate the E6 notebook: the transformation battery, done properly.

Reviewer S9 found the published table confounded. It ranks five rewrites by how
far macro-F1 falls when each is applied to a whole evaluation set, but the
rewrites fire at very different rates: renaming altered 100% of files, stripping
comments 81%, normalising whitespace 85%. A rewrite that fires more often moves
the aggregate more whatever its per-file effect, so the headline claim that
renaming costs most was never established.

The revision plan proposes dividing the aggregate delta by the application
rate. That is not valid: macro-F1 is an average of per-class ratios and does
not decompose as a weighted sum over rows, so the quotient is not the effect on
the altered subset. On a synthetic case with a constant true effect the
quotient errs by up to 0.216, and by 0.093 at exactly the 81% rate strip
comments has. A test in the suite records that counterexample.

The module now scores the baseline and the rewritten inputs on the same altered
rows, which is the only sound comparison. This run also answers the rest of the
finding: n rises from 2,000 to 10,000, the condition is named rather than left
implicit, and the sample is stratified by class.

    python kaggle_runs/make_e6_notebook.py
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_e1_notebooks as G  # noqa: E402

INTRO = """# E6: the transformation battery, reconditioned

The published table ranks semantics-preserving rewrites by how far macro-F1
falls when each is applied to a whole evaluation set. Reviewer S9 pointed out
that this is confounded, and it is: the rewrites fire at very different rates.

| Rewrite | Applied to |
|---|---|
| rename identifiers | 100.0% |
| normalise whitespace | 85.3% |
| strip comments | 81.0% |
| insert dead code | 100.0% |
| compress blank lines | 5.9% |

A rewrite that fires on more files moves the aggregate more whatever it does
per file, so the claim that renaming costs the most was never actually
established against stripping comments.

## What this run fixes

**The conditional effect, computed rather than derived.** The suite now scores
the baseline and the rewritten inputs on **the same altered rows**. The plan
suggested dividing the aggregate delta by the application rate; that is invalid,
because macro-F1 is an average of per-class ratios and does not decompose as a
weighted sum over rows. On a synthetic case with a constant true effect the
quotient errs by up to 0.216, and by 0.093 at exactly the 81% rate strip
comments has. The counterexample is pinned as a test.

**n = 10,000 rather than 2,000**, on a **named** condition, **stratified by
class**. All three were part of the same finding.

What this run does not yet cover, and the paper should not claim: parse-success
validation of the rewrites, AST-level structural rewrites, and running the
battery across the whole detector panel rather than the published detector
alone.

## Settings

| Setting | Value |
|---|---|
| **Accelerator** | `GPU T4 x2` |
| **Internet** | `On` |
| **Persistence** | `Files only` |

**Save Version -> Save & Run All (Commit).** Budget about 1.5 h: inference
only, six passes over 10,000 rows.
"""

RUN = '''run(["-m", "aicd.eval.transform_suite", "--config", CFG,
     "--model", "droiddetect", "--slice", "s1_in_distribution",
     "--n", "10000", "--batch-size", "32"])
elapsed("transformation battery")'''

REPORT = '''import json
rep = WORK / "aicd" / "eval" / "reports" / "transform_suite_droiddetect.json"
r = json.load(open(rep))
print(f"slice {r['slice']}, n = {r['n']:,}")
print(f"baseline macro-F1 {r['baseline']['macro_f1']:.4f}")
print()
print(f"{'rewrite':22s} {'applied':>8s} {'altered':>8s} {'aggregate':>10s} "
      f"{'conditional':>12s}")
print("-" * 66)
rows = sorted(r["transforms"], key=lambda x: x.get("delta_macro_f1_conditional") or 0)
for t in rows:
    cond = t.get("delta_macro_f1_conditional")
    cs = f"{cond:+12.4f}" if cond == cond and cond is not None else f"{'n/a':>12s}"
    print(f"{t['transform']:22s} {t['applied_fraction']:7.1%} "
          f"{t.get('n_altered', 0):>8,} {t['delta_macro_f1']:+10.4f} {cs}")
print()
print("Ranked by the conditional effect, which is what the rewrite costs on the")
print("files it actually touched. If that ordering differs from the aggregate")
print("ordering, the published ranking was an artefact of application rate and")
print("the paper's mechanism sentence has to be rewritten around this table.")'''

SAVE = '''OUT = pathlib.Path("/kaggle/working/results")
OUT.mkdir(parents=True, exist_ok=True)
reports = WORK / "aicd" / "eval" / "reports"
if reports.exists():
    shutil.copytree(reports, OUT / "reports", dirs_exist_ok=True)
shutil.make_archive("/kaggle/working/results", "zip", OUT)
print("-> /kaggle/working/results.zip")
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"  {p.stat().st_size/1024:8.0f} KB  {p.relative_to(OUT)}")
elapsed("saved")'''


def main() -> None:
    cells = [
        G.md(INTRO),
        G.code(G.SETUP),
        G.md("## 1. Code"), G.code(G.CODE),
        G.md("## 2. Dependencies"), G.code(G.DEPS),
        G.md("## 3. Resume, if a previous run is attached"),
        G.code(G.RESTORE.format(tag="e6")),
        G.md("## 4. Build the corpus"), G.code(G.BUILD),
        G.md("## 5. The battery, at n = 10,000"), G.code(RUN),
        G.md("## 6. Aggregate against conditional"), G.code(REPORT),
        G.md("## 7. Save"), G.code(SAVE),
    ]
    for c in cells:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    p = HERE / "11_e6_transforms.ipynb"
    p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {p.name}  ({len(cells)} cells, all code parses)")


if __name__ == "__main__":
    main()
