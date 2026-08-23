"""Generate the E7 notebook: score the published detector on CodeMirage.

Reviewer U6, raised by all three: AIGCodeSet is not a proper independent check.
It is Python only, it lacks a fourth class, and its human half descends from
Project CodeNet, the same upstream our own collection draws on.

CodeMirage differs on every one of those. Its human code comes from GitHub via
CodeParrot, sanitised in May 2022 before code LLMs were widely deployed. It
spans ten languages including Ruby and HTML, which DroidCollection contains
none of. And its generators fall into three tiers against our training data,
which is what makes it worth more than a second aggregate number.

This notebook scores the published DroidDetect-Base on it. That model is
loaded from its own repository, so the job needs no checkpoint of ours and
fits any free slot.

    python kaggle_runs/make_e7_notebook.py
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_e1_notebooks as G  # noqa: E402

INTRO = """# E7: the published detector on CodeMirage

Reviewer U6 was raised independently by all three reviewers: AIGCodeSet is not
properly independent. It is Python only, it has three of our four classes, and
its human code descends from Project CodeNet, which our own collection also
draws on. Calling it external overstates what it establishes.

CodeMirage (Guo et al., NeurIPS 2025) differs on every axis that mattered:

| | AIGCodeSet | CodeMirage |
|---|---|---|
| Human source | CodeNet, **shared upstream** | GitHub via CodeParrot, sanitised May 2022 |
| Languages | Python only | **ten**, incl. Ruby and HTML which DroidCollection has none of |
| Overlap with DroidCollection | 3 rows in 7,471 | **1 row in 62,995** |
| Fourth class | absent | paraphrased variants |

## What this run is actually for

Not another aggregate. CodeMirage's generators split three ways against our
training exposure, and the middle tier is the distinction this paper argues
benchmarks conflate:

| Tier | Generators | In DroidCollection? |
|---|---|---|
| `unseen_family` | Claude-3.5-Haiku, o3-mini | family absent entirely |
| `seen_family_unseen_model` | Gemini 2.0 x3, DeepSeek V3/R1 | family present, these models not |
| `seen_family_seen_model` | GPT-4o-mini, Llama-3.3-70B, Qwen2.5-Coder | present by name |

If macro-F1 falls across those tiers in order, the paper's central claim
reproduces on a corpus neither we nor the DroidCollection authors assembled,
and as a gradient rather than a binary. If it does not, that is equally worth
reporting, and the notebook says so rather than burying it.

The overlap audit is deliberately skipped here: it was measured once when the
corpus was built, at one row in 62,995, and repeating it would mean
downloading DroidCollection to re-derive a number we already have.

## Settings

| Setting | Value |
|---|---|
| **Accelerator** | `GPU T4 x2` |
| **Internet** | `On` (CodeMirage and DroidDetect both download) |
| **Persistence** | `Files only` |

**Save Version -> Save & Run All (Commit).** Budget about 2 h.
"""

BUILD = '''run(["-m", "aicd.eval.codemirage_corpus", "--split", "test",
     "--skip-overlap"])
elapsed("CodeMirage downloaded and mapped")

# The published detector's weights are public but are not part of the code
# dataset. The loader fetches them itself now; this only reports the outcome
# early, so a network problem surfaces here rather than after the corpus is
# built. The first attempt at this notebook died eighty-five seconds in for
# exactly that reason.
from huggingface_hub import hf_hub_download
w = hf_hub_download(repo_id="project-droid/DroidDetect-Base",
                    filename="pytorch_model.bin")
print("DroidDetect-Base weights available at", w)
elapsed("detector weights fetched")'''

SCORE = '''run(["-m", "aicd.eval.codemirage_eval", "--config", "kaggle.yaml",
     "--model", "base", "--batch-size", "32"])
elapsed("DroidDetect-Base scored on CodeMirage")'''

REPORT = '''import json, pathlib
rep = WORK / "aicd" / "eval" / "reports" / "codemirage_eval_base.json"
r = json.load(open(rep))
print(f"rows scored: {r['rows']:,}")
o = r["overall"]
print(f"overall  macro-F1 {o['macro_f1_present']:.4f}  "
      f"binary AUC {o['binary_auc']:.4f}  human FPR {o['human_fpr']:.4f}")
print()
print(f"{'tier':28s} {'n':>7s} {'macroF1':>9s} {'binAUC':>8s} {'humanFPR':>9s}")
print("-" * 66)
for t in ["seen_family_seen_model", "seen_family_unseen_model", "unseen_family"]:
    v = r["by_exposure"].get(t)
    if v:
        print(f"{t:28s} {v['n']:>7,} {v['macro_f1_present']:9.4f} "
              f"{v['binary_auc']:8.4f} {v['human_fpr']:9.4f}")
print()
print("per language:")
for k, v in sorted(r["by_language"].items()):
    print(f"  {k:12s} n={v['n']:>6,}  macro-F1 {v['macro_f1_present']:.4f}")'''

SAVE = '''OUT = pathlib.Path("/kaggle/working/results")
OUT.mkdir(parents=True, exist_ok=True)
reports = WORK / "aicd" / "eval" / "reports"
if reports.exists():
    shutil.copytree(reports, OUT / "reports", dirs_exist_ok=True)
art = WORK / "aicd" / "artifacts"
npy = OUT / "arrays"; npy.mkdir(exist_ok=True)
n = 0
for f in list(art.glob("proba_codemirage*.npy")) + list(art.glob("rows_codemirage*.parquet")):
    shutil.copy(f, npy / f.name); n += 1
shutil.make_archive("/kaggle/working/results", "zip", OUT)
print(f"copied {n} files -> /kaggle/working/results.zip")
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
        G.md("## 3. Download CodeMirage and map it onto our label scheme"),
        G.code(BUILD),
        G.md("## 4. Score the published DroidDetect-Base"), G.code(SCORE),
        G.md("## 5. The exposure gradient"), G.code(REPORT),
        G.md("## 6. Save"), G.code(SAVE),
    ]
    for c in cells:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    p = HERE / "10_e7_codemirage.ipynb"
    p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {p.name}  ({len(cells)} cells, all code parses)")


if __name__ == "__main__":
    main()
