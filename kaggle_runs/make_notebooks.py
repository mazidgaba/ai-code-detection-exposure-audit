"""Generate the two Kaggle notebooks for the revision experiments.

Written as a generator rather than hand-edited .ipynb files, for the same
reason kaggle/make_notebook.py is: raw notebook JSON is easy to corrupt and
impossible to review in a diff.

    python kaggle_runs/make_notebooks.py

Produces:
    kaggle_runs/1_matched_scale.ipynb
    kaggle_runs/2_seed_sweep.ipynb
"""
from __future__ import annotations

import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MD, PY = "markdown", "code"


# --------------------------------------------------------------- shared cells
SETUP = r'''
import os, sys, time, subprocess, shutil, pathlib, json

T0 = time.time()
def elapsed(label=""):
    m = (time.time() - T0) / 60
    print(f"[{m:6.1f} min] {label}", flush=True)

def run(cmd):
    """Run a pipeline stage and stop the notebook if it fails.

    Without the raise a failed stage prints a traceback and the next cell
    happily trains on whatever stale data is lying around.
    """
    print(">>", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([sys.executable, "-u", *[str(c) for c in cmd]])
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(str(c) for c in cmd)}")

print("Python", sys.version.split()[0])
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name}  {p.total_memory/1e9:.1f} GB")
else:
    raise SystemExit("No GPU. Set Accelerator to GPU T4 x2 in the right panel.")
'''

GETCODE = r'''
WORK = pathlib.Path("/kaggle/working/project")
WORK.mkdir(parents=True, exist_ok=True)

def find_aicd():
    root = pathlib.Path("/kaggle/input")
    if not root.exists():
        return None
    for cand in root.rglob("aicd"):
        if (cand / "config.py").exists() and (cand / "models").is_dir():
            return cand
    return None

src = find_aicd()
if src is None:
    raise SystemExit(
        "aicd/ not found under /kaggle/input.\n"
        "Add your code dataset: right panel -> Input -> Add Input -> Datasets,\n"
        "then search for the dataset you created from aicd-code.zip.")

dest = WORK / "aicd"
if dest.exists():
    shutil.rmtree(dest)
shutil.copytree(src, dest)
os.chdir(WORK)
sys.path.insert(0, str(WORK))
print("code ->", dest)
print("configs present:", sorted(p.name for p in (dest/"configs").glob("kaggle*.yaml")))
'''

DEPS = r'''
pkgs = ["xgboost", "tree-sitter", "tree-sitter-language-pack",
        "datasets", "shap", "pyyaml", "scikit-learn", "pyarrow"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
import importlib
for m in ["xgboost", "sklearn", "transformers", "datasets", "yaml"]:
    mod = importlib.import_module(m)
    print(f"  ok  {m:14s} {getattr(mod, '__version__', '')}")
elapsed("deps")
'''

SAVE = r'''
OUT = pathlib.Path("/kaggle/working/results")
OUT.mkdir(parents=True, exist_ok=True)

reports = WORK / "aicd" / "eval" / "reports"
if reports.exists():
    shutil.copytree(reports, OUT / "reports", dirs_exist_ok=True)

# The probability arrays are what the analysis modules re-read at home, and
# they are small. The model weights are hundreds of MB and are not needed to
# reproduce any number in the paper, so they stay behind.
art = WORK / "aicd" / "artifacts"
npy = OUT / "arrays"
npy.mkdir(exist_ok=True)
n = 0
for f in art.glob("proba_a*.npy"):
    shutil.copy(f, npy / f.name); n += 1
for f in art.glob("labels.parquet"):
    shutil.copy(f, npy / f.name)
if (art / "kaggle").exists():
    for f in (art / "kaggle").glob("*"):
        if f.is_file() and f.stat().st_size < 200e6:
            shutil.copy(f, npy / f.name); n += 1

shutil.make_archive("/kaggle/working/results", "zip", OUT)
print(f"copied {n} arrays")
print("-> /kaggle/working/results.zip  (download this from the Output tab)")
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"  {p.stat().st_size/1024:8.0f} KB  {p.relative_to(OUT)}")
elapsed("saved")
'''


def nb(cells):
    return {
        "cells": [
            {"cell_type": t, "metadata": {},
             **({"outputs": [], "execution_count": None} if t == PY else {}),
             "source": s.strip("\n").split("\n")}
            for t, s in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def fix_sources(doc):
    """Notebook JSON wants each line to keep its newline except the last."""
    for c in doc["cells"]:
        lines = c["source"]
        c["source"] = [l + "\n" for l in lines[:-1]] + lines[-1:]
    return doc


# ------------------------------------------------------- notebook 1: matched
MATCHED = [
(MD, """
# Experiment 1 — Matched-scale control

**This is the experiment that decides whether the paper is Q1.**

Branch A collapses from 0.898 to 0.238 macro-F1 under compound shift. It was
trained on 196,854 rows; the published detector it is compared against was
trained on roughly 847,000. Two explanations fit that, and the paper currently
cannot separate them:

1. the detector is sensitive to genuinely withheld categories, or
2. it was simply trained on less data.

This notebook trains the same model, with the same categories withheld, on all
three DroidCollection training shards instead of one. If it still collapses,
explanation 2 is excluded.

## Settings in the right-hand panel

| Setting | Value |
|---|---|
| **Accelerator** | `GPU T4 x2` |
| **Internet** | `On` |
| **Persistence** | `Files only` |

Then **Save Version -> Save & Run All (Commit)**. Do not use interactive Run
All: a browser disconnect kills an interactive session, while a committed run
continues on Kaggle's servers whether or not your laptop is awake.

**Budget: 8-10 hours.** Kaggle's session cap is 12 h.
"""),
(PY, SETUP),
(MD, "## 1. Code"),
(PY, GETCODE),
(MD, "## 2. Dependencies"),
(PY, DEPS),
(MD, """
## 3. Build the corpus from all three training shards

`--train-shards 3` is the only change from the original run. Downloading and
filtering roughly 847k rows takes about an hour.

The split-integrity tests run straight after and **must pass**. If a withheld
category leaks into training, every number downstream is fiction, and the whole
point of this experiment is that the withholding is real.
"""),
(PY, r'''
CFG = "kaggle_matched.yaml"

run(["-m", "aicd.data.download", "--config", CFG, "--train-shards", "3"])
elapsed("downloaded 3 shards")

for stage in ["normalize", "filter", "splits"]:
    run(["-m", f"aicd.data.{stage}", "--config", CFG])
    elapsed(stage)

run(["-m", "pytest", "aicd/tests/", "-q"])
elapsed("split-integrity tests passed")
'''),
(MD, """
## 4. Confirm the training set actually grew

The experiment is meaningless if this number is not much larger than the
196,854 rows of the original run. Check it before spending seven hours of GPU
on it.
"""),
(PY, r'''
import pandas as pd
sp = json.load(open(WORK / "aicd" / "eval" / "reports" / "splits.json"))
rows = sp["train"]["rows"]
print(f"training rows: {rows:,}   (original run: 196,854)")
print(f"ratio: {rows/196854:.2f}x")
if rows < 300000:
    raise SystemExit(
        f"Only {rows:,} training rows. Expected roughly 545,000.\\n"
        "The extra shards did not make it into the split. Check that cell 3\\n"
        "passed --train-shards 3 and that download.py did not skip cached\\n"
        "single-shard files; delete artifacts/data/hf/ and re-run if so.")
print("\\nper-class:", sp["train"]["labels"])
'''),
(MD, """
## 5. Train

`--tag matched` keeps this run's checkpoint, weights, report and probability
arrays separate from the original run's. Without a distinct tag the two
overwrite each other, and `--resume` would continue training the *other*
model.

`--resume` makes the cell safe to re-run: each epoch writes a checkpoint with
model, optimiser, scheduler and scaler state, so an interrupted run picks up at
the next epoch rather than starting over.

Roughly 6-8 hours for three epochs at this scale. If the session is killed,
re-run this one cell.
"""),
(PY, r'''
run(["-m", "aicd.models.modernbert_triplet", "--config", CFG,
     "--tag", "matched", "--resume"])
elapsed("branch A (matched scale) trained and evaluated")
'''),
(MD, "## 6. Read the result"),
(PY, r'''
rep = WORK / "aicd" / "eval" / "reports" / "branch_a_matched.json"
r = json.load(open(rep))["slices"]
print(f"{'condition':24s} {'matched':>9s} {'original':>9s}")
print("-" * 45)
ORIG = {"s1_in_distribution": 0.8977, "s2_unseen_generator": 0.8685,
        "s3_unseen_language": 0.5667, "s4_unseen_domain": 0.4029,
        "s5_compound": 0.2378}
for s, o in ORIG.items():
    if s in r:
        print(f"{s:24s} {r[s]['macro_f1']:9.4f} {o:9.4f}")
s1 = r["s1_in_distribution"]["macro_f1"]; s5 = r["s5_compound"]["macro_f1"]
print(f"\ncollapse S1 -> S5: {s1:.4f} -> {s5:.4f}  (drop {s1-s5:.4f})")
print(f"original run drop: {0.8977-0.2378:.4f}")
print()
if s5 < 0.45:
    print("COLLAPSE PERSISTS at roughly 3x the training data.")
    print("Data volume is excluded. The paper's causal claim holds.")
else:
    print("COLLAPSE DOES NOT PERSIST. This is a real finding and must be")
    print("reported: part of the original effect was a data-volume artefact,")
    print("and the paper's framing has to change to match.")
'''),
(MD, "## 7. Save"),
(PY, SAVE),
]

# ------------------------------------------------------- notebook 2: seeds
SEEDS = [
(MD, """
# Experiment 2 — Seed sweep

Bootstrap intervals in the paper resample the *evaluation set* for a model
whose weights are fixed. They say nothing about how much the result moves when
the same recipe is trained again from a different initialisation. A referee
will ask, and the honest answer today is that we do not know.

This trains two more models on **exactly the data of the original run**,
differing only in `project.seed`, and reports the spread.

The claim to support is undemanding: the S1-to-S5 gap is 0.66, so almost any
plausible seed-to-seed spread leaves it intact. This is cheap insurance rather
than a risk, which is why it is second priority behind the matched-scale run.

## Settings

Same as Experiment 1: `GPU T4 x2`, Internet `On`, and **Save & Run All
(Commit)** rather than interactive.

**Budget: 6-7 hours** (about an hour of data prep, then 2.5 h per seed).

> A bug worth knowing about, now fixed: `project.seed` used to reach only
> pandas sampling. Nothing seeded torch, so weight initialisation, dropout and
> batch order were left to default entropy. Before the fix this experiment
> would have run and produced numbers that varied for reasons unrelated to the
> seed.
"""),
(PY, SETUP),
(MD, "## 1. Code"),
(PY, GETCODE),
(MD, "## 2. Dependencies"),
(PY, DEPS),
(MD, """
## 3. Corpus

One shard, exactly as the original run, so these seeds are comparable with the
numbers already in the paper.
"""),
(PY, r'''
run(["-m", "aicd.data.download", "--config", "kaggle.yaml", "--train-shards", "1"])
for stage in ["normalize", "filter", "splits"]:
    run(["-m", f"aicd.data.{stage}", "--config", "kaggle.yaml"])
    elapsed(stage)

run(["-m", "pytest", "aicd/tests/", "-q"])

sp = json.load(open(WORK / "aicd" / "eval" / "reports" / "splits.json"))
print(f"training rows: {sp['train']['rows']:,}  (paper: 196,854)")
if abs(sp["train"]["rows"] - 196854) > 20000:
    print("WARNING: this does not match the published split. These seeds will")
    print("not be directly comparable with Table III.")
'''),
(MD, """
## 4. Train the two extra seeds

Each gets its own `--tag`, so the three runs cannot overwrite one another.

`--resume` is safe here *because* the tags differ: a resumed run continues its
own checkpoint. Re-run the cell after an interruption and it continues where it
stopped.
"""),
(PY, r'''
for seed in [1, 2]:
    run(["-m", "aicd.models.modernbert_triplet",
         "--config", f"kaggle_seed{seed}.yaml",
         "--tag", f"seed{seed}", "--resume"])
    elapsed(f"seed {seed} done")
'''),
(MD, "## 5. Spread across seeds"),
(PY, r'''
import numpy as np
rep = WORK / "aicd" / "eval" / "reports"

runs = {"paper (seed 20260818)": {"s1_in_distribution": 0.8977,
                                  "s5_compound": 0.2378}}
for seed in [1, 2]:
    f = rep / f"branch_a_seed{seed}.json"
    if f.exists():
        r = json.load(open(f))["slices"]
        runs[f"seed {seed}"] = {k: r[k]["macro_f1"] for k in
                                ["s1_in_distribution", "s5_compound"] if k in r}

print(f"{'run':24s} {'S1':>9s} {'S5':>9s}")
print("-" * 45)
for name, v in runs.items():
    print(f"{name:24s} {v.get('s1_in_distribution', float('nan')):9.4f} "
          f"{v.get('s5_compound', float('nan')):9.4f}")

s1 = [v["s1_in_distribution"] for v in runs.values() if "s1_in_distribution" in v]
s5 = [v["s5_compound"] for v in runs.values() if "s5_compound" in v]
if len(s1) > 1:
    print(f"\nS1  mean {np.mean(s1):.4f}  sd {np.std(s1, ddof=1):.4f}")
    print(f"S5  mean {np.mean(s5):.4f}  sd {np.std(s5, ddof=1):.4f}")
    gap = np.mean(s1) - np.mean(s5)
    sd = max(np.std(s1, ddof=1), np.std(s5, ddof=1))
    print(f"\ngap {gap:.4f} vs largest seed sd {sd:.4f}  ->  {gap/max(sd,1e-9):.0f}x")
    print("\nReport as: mean +/- sd over 3 seeds, in Table III.")
'''),
(MD, "## 6. Save"),
(PY, SAVE),
]


# ------------------------------------------------- notebook 3: resume matched
RESUME = [
(MD, """
# Experiment 1 (continued) — resume the matched-scale run

The first session hit Kaggle's 12-hour cap and was killed with exit code 137.
Nothing is lost. Checkpoints are written to
`/kaggle/working/project/aicd/artifacts/`, which is inside the notebook output,
so the epoch-1 checkpoint survived along with the built corpus.

This notebook picks up at epoch 2 and runs the evaluation. It needs about
**7 hours**, comfortably inside one session: roughly 4.9 h for the last epoch
and 2 h for evaluation.

Only two files matter for resuming:

| File | Why |
|---|---|
| `artifacts/branch_a_matched_ckpt.pt` | model, optimiser, scheduler, scaler, epoch |
| `artifacts/data/splits.parquet` | the training data |

The 647 MB of raw DroidCollection parquets are **not** needed, so nothing is
downloaded again.

## Settings

| Setting | Value |
|---|---|
| **Accelerator** | `GPU T4 x2` |
| **Internet** | `On` (the tokenizer and base weights still come from HuggingFace) |
| **Persistence** | `Files only` |

**Input → Add Input**, and add *both*:
1. **Datasets →** `aicd-code`
2. **Your Work / Notebooks →** the `Matched-scale` notebook, version 3

Then **Save Version → Save & Run All (Commit)**.
"""),
(PY, SETUP),
(MD, "## 1. Code"),
(PY, GETCODE),
(MD, "## 2. Dependencies"),
(PY, DEPS),
(MD, """
## 3. Restore the checkpoint and the corpus

Searches every attached input for the two files that matter. If the checkpoint
is missing this stops immediately rather than silently starting a fresh 15-hour
run, which is the one failure mode that would waste another whole quota.
"""),
(PY, r'''
art = WORK / "aicd" / "artifacts"
(art / "data").mkdir(parents=True, exist_ok=True)

def newest(pattern):
    hits = sorted(pathlib.Path("/kaggle/input").rglob(pattern),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None

ck_src = newest("branch_a_matched_ckpt.pt")
sp_src = newest("splits.parquet")

if ck_src is None:
    raise SystemExit(
        "branch_a_matched_ckpt.pt not found in any attached input.\n"
        "Add the previous run's output: Input -> Add Input -> Your Work ->\n"
        "the 'Matched-scale' notebook. Without it this notebook would start\n"
        "training from scratch and burn another 15 hours.")
if sp_src is None:
    raise SystemExit("splits.parquet not found. Attach the same notebook output.")

shutil.copy(ck_src, art / "branch_a_matched_ckpt.pt")
shutil.copy(sp_src, art / "data" / "splits.parquet")
print(f"checkpoint <- {ck_src}  ({ck_src.stat().st_size/1e9:.2f} GB)")
print(f"splits     <- {sp_src}  ({sp_src.stat().st_size/1e6:.0f} MB)")

# Read the epoch out of the checkpoint before training, so the plan is visible
# up front rather than inferred from the log an hour later.
ck = torch.load(art / "branch_a_matched_ckpt.pt", map_location="cpu",
                weights_only=False)
done = ck["epoch"] + 1
print(f"\ncompleted epochs : {done} of 3")
print(f"this session runs: epoch {done} to 2  ({3 - done} epoch(s))")
print(f"estimated        : {(3-done)*4.9:.1f} h training + ~2 h evaluation")
del ck
elapsed("restored")
'''),
(MD, """
## 4. Finish training, then evaluate

Same command as the first session. `--resume` finds the checkpoint and starts
at the next epoch; `--tag matched` keeps this run's artefacts separate from the
original single-shard run.
"""),
(PY, r'''
run(["-m", "aicd.models.modernbert_triplet", "--config", "kaggle_matched.yaml",
     "--tag", "matched", "--resume"])
elapsed("branch A (matched scale) finished")
'''),
(MD, "## 5. The verdict"),
(PY, r'''
r = json.load(open(WORK / "aicd" / "eval" / "reports" / "branch_a_matched.json"))["slices"]
ORIG = {"s1_in_distribution": 0.8977, "s2_unseen_generator": 0.8685,
        "s3_unseen_language": 0.5667, "s4_unseen_domain": 0.4029,
        "s5_compound": 0.2378}
print(f"{'condition':24s} {'matched':>9s} {'original':>9s}   n rows")
print("-" * 56)
for s, o in ORIG.items():
    if s in r:
        print(f"{s:24s} {r[s]['macro_f1']:9.4f} {o:9.4f}   {r[s].get('n', 0):,}")
s1, s5 = r["s1_in_distribution"]["macro_f1"], r["s5_compound"]["macro_f1"]
print(f"\nmatched  S1 -> S5: {s1:.4f} -> {s5:.4f}   drop {s1-s5:.4f}")
print(f"original S1 -> S5: 0.8977 -> 0.2378   drop {0.8977-0.2378:.4f}")
print("\ntrained on 394,624 rows against the original 196,854 (2.0x)\n")
if s5 < 0.45:
    print("COLLAPSE PERSISTS at twice the training data.")
    print("Training-set size is excluded as the explanation, and the causal")
    print("claim in the paper stands. This closes the confound in Section VIII.")
else:
    print("COLLAPSE DOES NOT PERSIST. Report this: part of the original effect")
    print("was a data-volume artefact and the framing must change. The exposure")
    print("audit and contamination results are unaffected either way.")
'''),
(MD, "## 6. Save"),
(PY, SAVE),
]


# --------------------------------------------- notebook 4: resume anything
RESUME_ANY = [
(MD, """
# Resume whatever is unfinished

Works for the matched-scale run, the seed sweep, or both, on whichever account
holds the quota. It looks at every checkpoint you attach, works out how far
each run got, and continues the ones that are incomplete.

Use this instead of notebook 3 when you are not certain what finished.

## Attach the inputs

**Input → Add Input**, and add:

1. Datasets → `aicd-code`
2. Every checkpoint source you have. Either the original notebook's output
   (Your Work → Notebooks) if it is on this account, or the checkpoint
   dataset you uploaded (see *Moving a checkpoint between accounts* in the
   README) if it is not.

Settings: `GPU T4 x2`, Internet `On`, then **Save Version → Save & Run All**.

## The 12-hour cap

`--max-hours` now stops training cleanly after the last epoch that fits,
instead of being killed partway through the next one. The first matched-scale
run lost 2.1 hours of GPU to a partial epoch that was thrown away; this stops
that happening again. Budget below is set to 9.5 h of training, leaving room
for evaluation inside a 12 h session.
"""),
(PY, SETUP),
(MD, "## 1. Code"),
(PY, GETCODE),
(MD, "## 2. Dependencies"),
(PY, DEPS),
(MD, """
## 3. Take stock

Finds every checkpoint and every corpus among the attached inputs, and reports
what each run still needs. Nothing trains until you have seen this table.
"""),
(PY, r'''
art = WORK / "aicd" / "artifacts"
(art / "data").mkdir(parents=True, exist_ok=True)
IN = pathlib.Path("/kaggle/input")

# A run is identified by its tag, which is embedded in the checkpoint name.
# Keep the newest copy if the same tag appears in more than one input.
found = {}
for p in IN.rglob("branch_a_*_ckpt.pt"):
    tag = p.name[len("branch_a_"):-len("_ckpt.pt")]
    if tag not in found or p.stat().st_mtime > found[tag].stat().st_mtime:
        found[tag] = p

# Each tag needs the corpus it was trained on. matched used three shards,
# the seeds used one, and the two are different files with the same name.
corpora = sorted(IN.rglob("splits.parquet"), key=lambda p: p.stat().st_size,
                 reverse=True)

print(f"checkpoints found: {len(found)}")
print(f"corpora found    : {len(corpora)}")
for c in corpora:
    print(f"    {c.stat().st_size/1e6:8.0f} MB  {c}")
print()

if not found:
    raise SystemExit(
        "No branch_a_*_ckpt.pt in any attached input.\n"
        "Attach the notebook output, or the checkpoint dataset, that holds it.")

TOTAL_EPOCHS = 3
plan = []
print(f"{'tag':12s} {'epochs done':>12s} {'remaining':>10s}  {'est. hours':>10s}")
print("-" * 50)
for tag, p in sorted(found.items()):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    done = ck["epoch"] + 1
    del ck
    left = TOTAL_EPOCHS - done
    # 4.9 h/epoch for matched (394,624 rows), 2.4 h/epoch for a seed (196,854).
    per = 4.9 if tag == "matched" else 2.4
    print(f"{tag:12s} {done:>7d} of {TOTAL_EPOCHS} {left:>10d}  {left*per:>10.1f}")
    if left > 0 or True:      # even a finished run still needs evaluation
        plan.append((tag, p, left, per))
print()
elapsed("stock taken")
'''),
(MD, """
## 4. Choose what to run this session

`matched` first: it is the experiment that decides the paper's tier. Edit
`ONLY` below if you want to force a particular run.
"""),
(PY, r'''
ONLY = None          # e.g. "matched" or "seed1"; None = everything found

order = {"matched": 0}
plan.sort(key=lambda t: order.get(t[0], 1))
todo = [t for t in plan if ONLY is None or t[0] == ONLY]

BUDGET_H = 9.5       # training hours; leaves ~2 h for evaluation under the cap

# Ask the trainer what it supports rather than assuming. An attached code
# dataset can be an older version than the one on your laptop, and passing a
# flag it has never heard of kills the run in the first two minutes.
_h = subprocess.run([sys.executable, "-m", "aicd.models.modernbert_triplet",
                     "--help"], capture_output=True, text=True)
HAS_MAX_HOURS = "--max-hours" in (_h.stdout + _h.stderr)
HAS_TAG = "--tag" in (_h.stdout + _h.stderr)

if not HAS_TAG:
    raise SystemExit(
        "The attached aicd-code dataset is too old: it has no --tag flag.\n"
        "Upload kaggle/aicd-code.zip as a New Version of that dataset, then\n"
        "in this notebook's Input panel switch it to the latest version.")
if not HAS_MAX_HOURS:
    print("NOTE: this code dataset predates --max-hours. Training will run")
    print("      without a clean stop, so a long run may be killed mid-epoch")
    print("      and lose that epoch's progress. Upload the current zip to")
    print("      fix. Continuing anyway; nothing is at risk.\n")

print("this session will run:")
for tag, _, left, per in todo:
    print(f"  {tag:12s} {left} epoch(s) left, about {left*per:.1f} h")
print(f"\ntraining budget: {BUDGET_H} h  (12 h session cap)")
if sum(l * p for _, _, l, p in todo) > BUDGET_H:
    print("\nThis exceeds the budget. --max-hours will stop cleanly after the")
    print("last epoch that fits, and the rest resumes next session.")
'''),
(MD, """
## 5. Restore and run

For each run: copy in its checkpoint and the matching corpus, then train with
`--resume`. The corpus is chosen by size — the matched build is roughly twice
the single-shard one — so the wrong data cannot be paired with a checkpoint.
"""),
(PY, r'''
for tag, ck_src, left, per in todo:
    print("\n" + "=" * 60)
    print(f"  {tag}")
    print("=" * 60, flush=True)

    shutil.copy(ck_src, art / f"branch_a_{tag}_ckpt.pt")
    print(f"checkpoint <- {ck_src.name}  ({ck_src.stat().st_size/1e9:.2f} GB)")

    # Largest corpus for matched, smallest for a seed run.
    src = corpora[0] if tag == "matched" else corpora[-1]
    shutil.copy(src, art / "data" / "splits.parquet")
    import pandas as pd
    n = len(pd.read_parquet(art / "data" / "splits.parquet", columns=["split"]))
    print(f"corpus     <- {src}  ({n:,} rows total)")

    cfg = "kaggle_matched.yaml" if tag == "matched" else f"kaggle_{tag}.yaml"
    cmd = ["-m", "aicd.models.modernbert_triplet", "--config", cfg,
           "--tag", tag, "--resume"]
    if HAS_MAX_HOURS:
        cmd += ["--max-hours", str(BUDGET_H)]
    run(cmd)
    elapsed(f"{tag} done for this session")
'''),
(MD, "## 6. Where each run stands now"),
(PY, r'''
rep = WORK / "aicd" / "eval" / "reports"
ORIG = {"s1_in_distribution": 0.8977, "s2_unseen_generator": 0.8685,
        "s3_unseen_language": 0.5667, "s4_unseen_domain": 0.4029,
        "s5_compound": 0.2378}

done_any = False
for f in sorted(rep.glob("branch_a_*.json")):
    tag = f.stem[len("branch_a_"):]
    r = json.load(open(f))["slices"]
    if "s5_compound" not in r:
        continue
    done_any = True
    s1, s5 = r["s1_in_distribution"]["macro_f1"], r["s5_compound"]["macro_f1"]
    print(f"\n{tag}:  S1 {s1:.4f} -> S5 {s5:.4f}   drop {s1-s5:.4f}")
    if tag == "matched":
        print(f"  original (196,854 rows): 0.8977 -> 0.2378   drop 0.6599")
        print(f"  matched  (394,624 rows), a factor of 2.0")
        if s5 < 0.45:
            print("  COLLAPSE PERSISTS. Training-set size is excluded.")
        else:
            print("  COLLAPSE DOES NOT PERSIST. Report it; the framing changes.")

if not done_any:
    print("No run finished all 3 epochs yet. Checkpoints are saved.")
    print("Re-run this notebook next session and it continues from here.")
'''),
(MD, "## 7. Save"),
(PY, SAVE),
]


def main() -> None:
    for name, cells in [("1_matched_scale", MATCHED), ("2_seed_sweep", SEEDS),
                        ("3_resume_matched", RESUME),
                        ("4_resume_any", RESUME_ANY)]:
        p = os.path.join(HERE, name + ".ipynb")
        doc = fix_sources(nb(cells))
        io.open(p, "w", encoding="utf-8", newline="\n").write(
            json.dumps(doc, indent=1))
        print(f"  {os.path.getsize(p)/1024:6.1f} KB  {name}.ipynb")


if __name__ == "__main__":
    main()
