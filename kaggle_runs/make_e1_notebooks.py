"""Generate the two E1 notebooks: the exposed arm and its unexposed twin.

Both build the same corpus from the same config and the same seed, so the arm
partition is identical on both accounts without either needing to see the
other's output. Each then trains one arm. They can run at the same time on two
Kaggle accounts, and neither depends on the other finishing.

Each notebook prints a checksum of the arm assignment. The two runs must agree
on it. If they do not, the corpus was built differently and the comparison is
void, so the number is printed prominently rather than buried in a log.

    python kaggle_runs/make_e1_notebooks.py
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

ARMS = {
    "d2": {
        "n": 5,
        "title": "E1a: the exposed arm (D2)",
        "tag": "e1_d2",
        "hours": "10-12 h",
        "what": """Trains a detector on 196,854 rows of which 35.6% carry a category the
benchmark withholds. That fraction is not arbitrary: it is what the exposure
audit measured in the training split the published detector names, so this arm
is exposed to roughly what that detector was exposed to.""",
    },
    "d1small": {
        "n": 6,
        "title": "E1b: the unexposed twin (D1_small)",
        "tag": "e1_d1small",
        "hours": "8-10 h",
        "what": """Trains on exactly D2's retained rows and nothing else: same rows, same
recipe, same seed, no withheld category present. This is the arm that makes the
comparison clean. D2 minus this is the exposure effect with the retained data
held constant, which is not true of D2 minus the original model.""",
    },
}

INTRO = """# {title}

{what}

## Why there are two notebooks

Holding the row count fixed while adding withheld rows necessarily removes
retained ones, so comparing the exposed arm against the original 196,854-row
model would confound exposure with dilution. Three arms separate the two:

| Arm | Rows | Withheld present |
|---|---|---|
| D1 (already trained) | 196,854 | none |
| **D2** | 196,854 | 35.6% |
| **D1_small** | ~126,800 | none |

`D2 - D1_small` is exposure with the retained data held constant.
`D1 - D1_small` is scale at constant composition. `D2 - D1` is what a
benchmark-relative evaluation would report at equal budget.

Donor rows are taken only from S2, S3 and S4. **S1 and S5 are untouched**, so
the existing model's saved probabilities remain directly comparable on them
with no rescoring.

## Settings in the right-hand panel

| Setting | Value |
|---|---|
| **Accelerator** | `GPU T4 x2` |
| **Internet** | `On` |
| **Persistence** | `Files only` |

Then **Save Version -> Save & Run All (Commit)**. Not interactive Run All: a
browser disconnect kills an interactive session, while a committed run
continues on Kaggle's servers.

**Budget: {hours}.** The session cap is 12 h and the trainer is given
`--max-hours 10`, so it stops cleanly after the last epoch that fits rather
than being killed part-way through one. Re-run this same notebook with the
previous version's output attached to continue.
"""

SETUP = '''import os, sys, time, subprocess, shutil, pathlib, json

T0 = time.time()
def elapsed(label=""):
    m = (time.time() - T0) / 60
    print(f"[{m:6.1f} min] {label}", flush=True)

def run(cmd):
    """Run a stage and stop the notebook if it fails, rather than letting the
    next cell train on whatever stale data is lying around."""
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
    raise SystemExit("No GPU. Set Accelerator to GPU T4 x2 in the right panel.")'''

CODE = '''WORK = pathlib.Path("/kaggle/working/project")
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
        "aicd/ not found under /kaggle/input.\\n"
        "Right panel -> Input -> Add Input -> Datasets, then add the dataset\\n"
        "you created from aicd-code.zip.")

dest = WORK / "aicd"
if dest.exists():
    shutil.rmtree(dest)
shutil.copytree(src, dest)
os.chdir(WORK)
sys.path.insert(0, str(WORK))
print("code ->", dest)
if not (dest / "data" / "exposure_arms.py").exists():
    raise SystemExit("This code dataset predates E1. Re-upload aicd-code.zip.")'''

DEPS = '''pkgs = ["xgboost", "tree-sitter", "tree-sitter-language-pack",
        "datasets", "shap", "pyyaml", "scikit-learn", "pyarrow", "datasketch"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
import importlib
for m in ["xgboost", "sklearn", "transformers", "datasets", "yaml", "datasketch"]:
    mod = importlib.import_module(m)
    print(f"  ok  {m:14s} {getattr(mod, '__version__', '')}")
elapsed("deps")'''

RESTORE = """# Resume support. A fresh Kaggle session starts with an empty working
# directory, so `--resume` alone finds nothing, prints "starting fresh" and
# silently retrains from epoch 0. That is how seven hours disappear without
# anyone noticing, so the checkpoint is restored explicitly here.
#
# The corpus is restored too, and that matters more than it looks. Rebuilding
# is deterministic given the seed but NOT across library versions: the same
# pipeline produced 417,645 rows earlier and 417,431 today, a parse-filter
# difference. Resuming a checkpoint onto a corpus it was not trained on would
# be quietly wrong, so if a previous split is available we use it and skip the
# rebuild entirely.
#
# Attach the previous version's output: Input -> Add Input -> Notebook Output.
# Set True when this notebook is pushed as a RESUME. A resume that silently
# finds nothing and trains from scratch is the expensive failure: it looks like
# a normal run and costs a full session. With this on, the notebook refuses.
REQUIRE_RESUME = False

import torch
art = WORK / "aicd" / "artifacts"
(art / "data").mkdir(parents=True, exist_ok=True)
CKPT = "branch_a_{tag}_ckpt.pt"

def newest(pattern):
    hits = sorted(pathlib.Path("/kaggle/input").rglob(pattern),
                  key=lambda q: q.stat().st_mtime, reverse=True)
    return hits[0] if hits else None

sp = newest("splits.parquet")
RESTORED_SPLITS = False
if sp is not None:
    shutil.copy(sp, art / "data" / "splits.parquet")
    import pandas as _pd
    _n = len(_pd.read_parquet(art / "data" / "splits.parquet", columns=["label"]))
    print(f"restored splits.parquet from {{sp}}  ({{_n:,}} rows)")
    RESTORED_SPLITS = True
else:
    print("no previous splits.parquet found; the corpus will be built fresh")

ck = newest(CKPT)
if ck is None:
    if REQUIRE_RESUME:
        raise SystemExit(
            f"This notebook was pushed as a RESUME but no {{CKPT}} was found "
            "under /kaggle/input. Training from scratch here would waste a "
            "whole session and look like success. Attach the previous "
            "version's output (Input -> Add Input -> Notebook Output) and "
            "re-run.")
    print(f"no {{CKPT}} under /kaggle/input -- this will train from epoch 0.")
    print("If you meant to resume, attach the previous version's output.")
else:
    shutil.copy(ck, art / CKPT)
    _c = torch.load(art / CKPT, map_location="cpu", weights_only=False)
    print(f"restored {{CKPT}} from {{ck}}")
    print(f"  holds epoch {{_c['epoch']}}, so training resumes at epoch {{_c['epoch'] + 1}}")
    if not RESTORED_SPLITS:
        raise SystemExit(
            "A checkpoint was restored but its corpus was not. Rebuilding may "
            "produce a different split than the one this checkpoint was "
            "trained on, which would make the resumed run unsound. Attach the "
            "previous output so splits.parquet comes with it.")
"""

BUILD = '''CFG = "kaggle.yaml"

# One train shard, not three. One shard plus dev and test is 493,850 raw rows
# which filter to the 417,645 of the original GPU build, with 196,854 of them
# training. Three shards is the matched-scale corpus and gives roughly 545,000
# training rows, which is a different experiment.
if RESTORED_SPLITS:
    print("corpus restored from the previous run; skipping the rebuild so the")
    print("resumed model continues on exactly the data it was trained on.")
    import pandas as pd
    _sp = pd.read_parquet(WORK / "aicd" / "artifacts" / "data" / "splits.parquet",
                          columns=["split"])
    rows = int((_sp["split"] == "train").sum())
    print(f"training rows: {rows:,}")
else:
    run(["-m", "aicd.data.download", "--config", CFG, "--train-shards", "1"])
    elapsed("downloaded 1 shard")

    for stage in ["normalize", "filter", "splits"]:
        run(["-m", f"aicd.data.{stage}", "--config", CFG])
        elapsed(stage)

    run(["-m", "pytest", "aicd/tests/", "-q"])
    elapsed("integrity tests passed")

    sp_report = json.load(open(WORK / "aicd" / "eval" / "reports" / "splits.json"))
    rows = sp_report["train"]["rows"]
    print(f"\\ntraining rows: {rows:,}   (expected 196,854)")
    if abs(rows - 196854) > 5000:
        raise SystemExit(
            f"Got {rows:,} training rows, expected about 196,854. This notebook "
            "must reproduce the original GPU build exactly, or the arms are not "
            "comparable with the existing model.")'''

DUP = '''# Phase 1 task 1.4, on the build that actually produces the paper's numbers.
# The corpus is already here and this needs no GPU, so it costs a few CPU
# minutes inside a session that is running anyway.
run(["-m", "aicd.eval.duplication_audit"])
elapsed("near-duplicate audit")'''

ARMBUILD = '''run(["-m", "aicd.data.exposure_arms", "--config", CFG, "--exposure", "0.356"])
elapsed("arms built")

import pandas as pd, hashlib
d = WORK / "aicd" / "artifacts" / "data"
arms = pd.read_parquet(d / "splits_arms.parquet",
                       columns=["d2_train", "d1small_train", "arm_slice"])
sig = hashlib.sha1(
    arms["d2_train"].to_numpy().tobytes()
    + arms["d1small_train"].to_numpy().tobytes()).hexdigest()[:16]

print("\\n" + "=" * 66)
print("ARM CHECKSUM:", sig)
print("=" * 66)
print("Both E1 notebooks must print the SAME checksum. If they differ, the")
print("two accounts built different partitions and D2 minus D1_small is not")
print("a comparison. Stop and re-check the config and seed before trusting")
print("any number from these runs.")
print()
print(json.dumps(json.load(open(d / "arm_report.json")), indent=2)[:1200])'''

TRAIN = '''run(["-m", "aicd.models.modernbert_triplet", "--config", CFG,
     "--arm", "{arm}", "--tag", "{tag}", "--max-hours", "10", "--resume"])
elapsed("{tag} trained and evaluated")'''

RESULT = '''rep = WORK / "aicd" / "eval" / "reports" / "branch_a_{tag}.json"
r = json.load(open(rep))["slices"]
print(f"{{'condition':24s}} {{'{tag}':>10s}}")
print("-" * 36)
for s in ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
          "s4_unseen_domain", "s5_compound"]:
    if s in r:
        print(f"{{s:24s}} {{r[s]['macro_f1']:10.4f}}")
s1 = r.get("s1_in_distribution", {{}}).get("macro_f1")
s5 = r.get("s5_compound", {{}}).get("macro_f1")
if s1 and s5:
    print(f"\\nS1 -> S5: {{s1:.4f}} -> {{s5:.4f}}  (drop {{s1-s5:.4f}})")
print("\\nThe comparison this arm belongs to can only be made once BOTH arms")
print("have run. Download results.zip and the analysis happens at home.")'''

SAVE = '''OUT = pathlib.Path("/kaggle/working/results")
OUT.mkdir(parents=True, exist_ok=True)

reports = WORK / "aicd" / "eval" / "reports"
if reports.exists():
    shutil.copytree(reports, OUT / "reports", dirs_exist_ok=True)

art = WORK / "aicd" / "artifacts"
npy = OUT / "arrays"
npy.mkdir(exist_ok=True)
n = 0
for f in art.glob("proba_a*.npy"):
    shutil.copy(f, npy / f.name); n += 1
for name in ("arm_report.json", "splits_arms.parquet"):
    p = art / "data" / name
    if p.exists() and p.stat().st_size < 200e6:
        shutil.copy(p, OUT / name)
masks = art / "data" / "arm_masks"
if masks.exists():
    shutil.copytree(masks, OUT / "arm_masks", dirs_exist_ok=True)
if (art / "kaggle").exists():
    for f in (art / "kaggle").glob("*"):
        if f.is_file() and f.stat().st_size < 200e6:
            shutil.copy(f, npy / f.name); n += 1

shutil.make_archive("/kaggle/working/results", "zip", OUT)
print(f"copied {n} arrays")
print("-> /kaggle/working/results.zip  (Output tab)")
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"  {p.stat().st_size/1024:8.0f} KB  {p.relative_to(OUT)}")
elapsed("saved")'''


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def build(arm: str, spec: dict) -> dict:
    cells = [
        md(INTRO.format(**spec)),
        code(SETUP),
        md("## 1. Code"),
        code(CODE),
        md("## 2. Dependencies"),
        code(DEPS),
        md("## 3. Resume, if a previous run is attached"), code(RESTORE.format(tag=spec["tag"])),
        md("## 3. Build the corpus\n\nThis reproduces the original GPU build "
           "exactly. Roughly an hour."),
        code(BUILD),
        md("## 4. Near-duplicate audit\n\nMeasures how much of each condition "
           "is a near-copy of something in training. Free, and it answers a "
           "question every referee asks."),
        code(DUP),
        md("## 5. Build the three arms\n\nDeterministic given the seed, which "
           "is what lets two accounts build the same partition independently."),
        code(ARMBUILD),
        md(f"## 6. Train {spec['tag']}\n\nCheckpoints every epoch. If the "
           "session dies, re-run this notebook with the previous output "
           "attached and `--resume` picks up where it stopped."),
        code(TRAIN.format(arm=arm, tag=spec["tag"])),
        md("## 7. This arm's numbers"),
        code(RESULT.format(tag=spec["tag"])),
        md("## 8. Save"),
        code(SAVE),
    ]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


def main() -> None:
    for arm, spec in ARMS.items():
        nb = build(arm, spec)
        p = HERE / f"{spec['n']}_e1_{arm}.ipynb"
        p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
