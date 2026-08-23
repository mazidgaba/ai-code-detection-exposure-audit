"""Generate the E9 notebook: Fast-DetectGPT, the zero-shot control.

Reviewer note: DetectGPT and Fast-DetectGPT are discussed in Related Work and
never evaluated. That is true, and the gap matters more than a missing baseline
usually does.

The paper argues that the S1-to-S5 collapse is caused by withholding. The
obvious competing explanation is that S3 to S5 are simply harder conditions:
different languages, different source domains, harder problems. A trained
detector cannot separate those, because it has both properties at once.

Fast-DetectGPT has no training set. It never saw DroidCollection, so nothing
was withheld from it, so whatever profile it traces across the five conditions
belongs to the conditions themselves:

  flat profile      the conditions are not intrinsically harder, and the
                    trained detector's collapse is about exposure. This is the
                    twin control's conclusion reached from the opposite side.
  matching collapse part of the effect is intrinsic difficulty, and the paper
                    must narrow its claim to say so.

Either outcome is reportable and neither is decoration.

A previous attempt ran on CPU at 400 rows per slice and died partway through
S4, leaving no report. Three defects made that inevitable and are fixed:
one-sample-at-a-time scoring, bf16 on a Turing card that has no hardware bf16,
and a dense vocabulary reduction costing about 1.9 GB per sample.

    python kaggle_runs/make_e9_notebook.py
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_e1_notebooks as G  # noqa: E402

INTRO = """# E9: Fast-DetectGPT, the zero-shot control

DetectGPT and Fast-DetectGPT are cited in Related Work and never evaluated.
This run closes that, and it is not only a missing-baseline fix.

## What it controls for

The paper's claim is that the collapse from S1 to S5 is caused by **withholding**.
A referee's natural objection is that S3 to S5 are just **harder** conditions.
A trained detector cannot distinguish those two explanations, because both are
true of it at once.

Fast-DetectGPT has **no training set at all**. Nothing was withheld from it, so
its profile across the five conditions is a property of the conditions:

| If the zero-shot profile is | Then |
|---|---|
| roughly **flat** | the conditions are not intrinsically harder; the collapse is exposure |
| a **matching collapse** | part of the effect is intrinsic difficulty and the paper narrows |

The trained detector's binary AUC on the same conditions is already stored, so
the comparison is like for like.

## Why the earlier attempt failed

Branch C ran once, on CPU, at 400 rows per slice, and died partway through S4.
Three defects, all fixed:

| Defect | Effect | Fix |
|---|---|---|
| scored one sample at a time | `batch_size` in the config was ignored | length-bucketed batching |
| bf16 on CUDA | a T4 is Turing and has no hardware bf16 | fp16 below sm_80 |
| dense vocabulary reduction | ~1.9 GB per sample at 512 tokens | chunked, ~0.5 GB per batch |

The batched path is asserted equal to the single-sample path, which is itself
tied to a closed-form oracle. Thirteen tests cover it.

## The probe, and what version 1 measured

Cell 5 scores 400 rows and projects the run before committing to it. Version 1
measured **26.0 rows/s** on a T4, projected 142 min for all 220,670 rows, and
stopped itself ten minutes in rather than spending the session finding out. The
FLOP arithmetic behind the original estimate was optimistic by roughly half;
the throughput this project actually gets on a T4, 21.6 rows/s in E7 and 31 to
33 in the E5-large evaluation, was the better guide.

Scoring every row was over-scoped regardless. This run needs a per-slice binary
AUC, and at **15,000 rows per slice** the 95% interval on an AUC near 0.7 is
about +/-0.007, far finer than any difference the conclusion turns on. That cap
gives 79,168 rows and about 51 minutes at the measured rate.

Version 1 also exposed a defect in the guard itself: it compared the capped
run against the uncapped row count, so it would have aborted a run that fits.
The projection now counts what will actually be scored.

## Settings

| Setting | Value |
|---|---|
| **Accelerator** | `GPU T4 x2` |
| **Internet** | `On` (pulls the scoring model) |
| **Persistence** | `Files only` |

**Save Version -> Save & Run All (Commit).** Budget about 65 min: 8.6 min of
corpus build, ~4 min for the scoring model, ~51 min of inference.
"""

PROBE = '''# Measure before committing. The full evaluation set is 220,670 rows; if the
# achieved rate makes that longer than the budget, stop now and say so rather
# than discovering it at hour eight.
import numpy as np, pandas as pd, torch, time
sys.path.insert(0, str(WORK))
from aicd import config as C
from aicd.models import fastdetect as FD

BUDGET_MIN = 100
PROBE_ROWS = 400

# Version 1 measured 26.0 rows/s on a T4 and projected 142 min for all 220,670
# rows, so the guard stopped it. Scoring every row was over-scoped anyway: this
# run needs a per-slice binary AUC, and at 15,000 rows the 95% interval on an
# AUC near 0.7 is about +/-0.007, which is far finer than any difference the
# result turns on. S5 has only 4,168 rows and is unaffected by the cap.
MAX_PER_SLICE = 15000

cfg = C.load(CFG)
splits = pd.read_parquet(WORK / "aicd" / "artifacts" / "data" / "splits.parquet",
                         columns=["slice", "label", "code"])
per = splits[splits["slice"].isin(["val"] + list(FD.SLICES))]["slice"].value_counts()
# The projection has to count what will actually be scored, not what exists.
# Version 1 measured the capped run against the uncapped total and would have
# aborted a run that fits comfortably.
TOTAL = int(sum(min(int(n), MAX_PER_SLICE) for n in per))
print(f"rows available: {int(per.sum()):,}   to score at a "
      f"{MAX_PER_SLICE:,}/slice cap: {TOTAL:,}")
for s, n in per.items():
    print(f"    {s:24s} {int(n):>7,} -> {min(int(n), MAX_PER_SLICE):>7,}")

tok, model, dev = FD.load_scorer(cfg)
print(f"scorer {cfg.fastdetect.scoring_model}  dev {dev}  dtype {FD._dtype_for(dev)}")

probe = splits[splits["slice"] == "s1_in_distribution"].sample(
    n=PROBE_ROWS, random_state=0)
t0 = time.time()
sc = FD.score_frame(probe, tok, model, dev, cfg, tag="probe")
dt = time.time() - t0

rate = PROBE_ROWS / dt
eta_min = TOTAL / rate / 60
print(f"\\nprobe: {PROBE_ROWS} rows in {dt:.1f}s = {rate:.1f} rows/s")
print(f"projected full run: {eta_min:.0f} min ({eta_min/60:.2f} h)")
print(f"scores: mean {sc.mean():+.3f}  sd {sc.std():.3f}  "
      f"zeros {(sc == 0).sum()}/{len(sc)}")

if (sc == 0).mean() > 0.10:
    raise SystemExit("More than a tenth of probe scores are exactly zero. That "
                     "is the declined-or-failed path, not a real score, and it "
                     "would read as confident human predictions.")
if eta_min > BUDGET_MIN:
    raise SystemExit(
        f"Projected {eta_min:.0f} min exceeds the {BUDGET_MIN} min budget. "
        f"Stopping before the session is spent. Re-run with --max-per-slice, "
        f"or raise the batch size if memory allows.")
print(f"\\nWithin budget ({eta_min:.0f} < {BUDGET_MIN} min). Proceeding.")
del model, tok
torch.cuda.empty_cache()
elapsed("probe")'''

RUN = '''run(["-m", "aicd.models.fastdetect", "--config", CFG,
     "--tag", "c", "--batch-size", "16", "--chunk", "256",
     "--max-per-slice", str(MAX_PER_SLICE)])
elapsed("Fast-DetectGPT scored")'''

REPORT = '''import json
rep = WORK / "aicd" / "eval" / "reports" / "branch_c_fastdetect.json"
r = json.load(open(rep))
print(f"scorer: {r['scorer']}\\n")
print(f"{'condition':24s} {'n':>8s} {'binAUC':>8s} {'mean d human':>13s} {'machine':>9s}")
print("-" * 68)
for s, v in r["slices"].items():
    h = v.get("mean_d_human"); m = v.get("mean_d_machine")
    print(f"{s:24s} {v['n']:>8,} {v['binary_auc']:8.4f} "
          f"{(h if h is not None else float('nan')):13.3f} "
          f"{(m if m is not None else float('nan')):9.3f}")

# The trained detectors' reports are outputs of earlier runs and are not shipped
# in the code dataset, so the comparison is made locally by
# aicd.eval.zeroshot_compare once these numbers are downloaded. What matters
# here is only that the run produced a usable profile, so that is what is
# checked, and the yardstick is stated rather than computed against a file that
# is not present.
z = r["slices"]
if "s1_in_distribution" in z and "s5_compound" in z:
    zd = z["s1_in_distribution"]["binary_auc"] - z["s5_compound"]["binary_auc"]
    print(f"\\nzero-shot S1 to S5 binary-AUC drop: {zd:+.4f}")
    print()
    print("For scale, from E1's twin control on the same conditions:")
    print("  unexposed arm D1_small   drop 0.2317")
    print("  exposed   arm D2         drop 0.0079")
    print()
    print("A zero-shot drop far below 0.2317 says the conditions are not")
    print("intrinsically harder, so the unexposed arm's decline is about what")
    print("was withheld. A drop near or above it says part of the effect is")
    print("intrinsic difficulty and the paper must narrow its claim.")

nz = sum(1 for v in z.values() if v["binary_auc"] == v["binary_auc"])
if nz < 6:
    raise SystemExit(f"only {nz} of 6 conditions produced an AUC; the earlier "
                     "attempt died partway through S4 and left no report, so "
                     "an incomplete run is failed here rather than saved.")'''

SAVE = '''OUT = pathlib.Path("/kaggle/working/results")
OUT.mkdir(parents=True, exist_ok=True)
reports = WORK / "aicd" / "eval" / "reports"
if reports.exists():
    shutil.copytree(reports, OUT / "reports", dirs_exist_ok=True)
art = WORK / "aicd" / "artifacts"
(OUT / "arrays").mkdir(exist_ok=True)
n = 0
for p in art.glob("score_c_*"):
    shutil.copy2(p, OUT / "arrays" / p.name)
    n += 1
print(f"copied {n} score files")
shutil.make_archive("/kaggle/working/results", "zip", OUT)
print("-> /kaggle/working/results.zip  (Output tab)")
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
        G.code(G.RESTORE.format(tag="e9")),
        G.md("## 4. Build the corpus"), G.code(G.BUILD),
        G.md("## 5. Throughput probe, and the budget guard"), G.code(PROBE),
        G.md("## 6. Score every row"), G.code(RUN),
        G.md("## 7. Zero-shot against trained"), G.code(REPORT),
        G.md("## 8. Save"), G.code(SAVE),
    ]
    for c in cells:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    p = HERE / "12_e9_fastdetect.ipynb"
    p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {p.name}  ({len(cells)} cells, all code parses)")


if __name__ == "__main__":
    main()
