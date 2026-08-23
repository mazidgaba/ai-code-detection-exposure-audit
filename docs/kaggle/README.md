# Kaggle runs — exactly what to do

Two experiments, in priority order. You have roughly 15 GPU hours left this
week; these need about 15. If you can only do one, **do Experiment 1** — it is
the one a referee will demand and the one that decides the paper's tier.

| | Experiment | Time (measured) | Sessions | Why it matters |
|---|---|---|---|---|
| 1 | Matched-scale control | up to 17.5 h | 2 | Decides whether the collapse is shift sensitivity or just less data |
| 1b | Resume after the 12 h cap | ~7 h | — | Finishes epoch 2 and evaluation |
| 2 | Seed sweep | up to 8.6 h per seed | 1 per seed | Shows the collapse is not training noise |

These come from the rate the real run produced, 1.42 s/step at batch 32 on one
T4, not from an estimate. Neither experiment fits in a single 12-hour session.

Everything in this folder is ready to upload. You should not need to edit any
code.

---

## Step 0 — Upload the code (once, ~5 minutes)

The zip has already been rebuilt and **contains the three new config files** the
notebooks need. If you re-run `python kaggle/prepare_upload.py` later, it will
pick them up automatically.

1. Open <https://www.kaggle.com/datasets> and click **New Dataset**.
2. Drag in `kaggle/aicd-code.zip` (118 KB).
3. Title it `aicd-code`. Click **Create**.

If you already have an `aicd-code` dataset from yesterday, **you must update
it** — the old one does not contain the new configs or the `--tag` fix. Open
it, click **New Version**, upload the new zip.

> Check after upload: the dataset should contain
> `aicd/configs/kaggle_matched.yaml`. If it does not, the notebooks will stop
> at cell 3 with a clear error rather than wasting GPU time.

---

## Step 1 — Experiment 1: matched-scale control

1. Go to <https://www.kaggle.com/code> and click **New Notebook**.
2. **File → Import Notebook**, upload `kaggle_runs/1_matched_scale.ipynb`.
3. Right-hand panel:
   - **Accelerator**: `GPU T4 x2`
   - **Internet**: `On`
   - **Persistence**: `Files only`
   - **Input → Add Input → Datasets →** search `aicd-code` → **Add**
4. Click **Save Version** (top right) → choose **Save & Run All (Commit)** →
   **Save**.
5. Close the tab. Really.

### Why "Save & Run All" and not the Run button

An interactive session belongs to your browser. Close the laptop, lose the
network, let the machine sleep, and the session dies with whatever it was
doing. A committed run executes on Kaggle's servers independently of you. This
is the single most important instruction on this page, and it is the reason
yesterday's hours are worth protecting.

You can watch progress under the notebook's **Version** tab, or just come back
in the morning.

### What it does

Downloads all three DroidCollection training shards instead of one, applies the
identical withholding, and retrains Branch A on 394,624 rows against the
original 196,854, a factor of 2.0. (The pre-run estimate was 545,000; the
filter removes more than expected at this scale, which the run measured.)

Cell 4 stops the run if the training set did not actually grow. That check
exists because `download.py` skips files it already has: if a cached
single-shard download were reused, the notebook would train the same small
model again and look like it worked.

Cell 6 prints the verdict directly:

```
collapse S1 -> S5: 0.xxxx -> 0.xxxx  (drop 0.xxxx)
original run drop: 0.6599
```

- **S5 still below ~0.45** → the collapse survives 2x the data. Volume is
  excluded, the paper's causal claim holds, and the confound in Section VIII
  is closed.
- **S5 well above ~0.45** → part of the original effect was a data-volume
  artefact. That is a real result and must be reported. It does not sink the
  paper — the contamination and exposure findings stand on their own — but the
  framing has to change, and I would rather you learn it from your own run than
  from a referee.

---

## Step 1b — If Experiment 1 hits the 12-hour cap

It will. Measured on the real run: 1.42 s/step, 12,332 steps per epoch, so
three epochs need 14.6 h of training plus about 2 h of evaluation. Kaggle kills
GPU notebooks at 12 h with exit code 137.

That is expected, not a failure. Epoch checkpoints land in
`/kaggle/working/project/aicd/artifacts/`, which is part of the notebook
output, so the run is resumable.

Timeline of the first session:

| Elapsed | Event |
|---|---|
| 5.0 h | epoch 0 done, checkpoint written |
| 9.9 h | epoch 1 done, checkpoint written |
| 12.0 h | killed roughly 43% into epoch 2 |

To finish, import `kaggle_runs/3_resume_matched.ipynb` and attach **two**
inputs: the `aicd-code` dataset *and* the killed notebook's output (Add Input →
Your Work → Notebooks). It restores the checkpoint and `splits.parquet`, runs
the final epoch and evaluates. About **7 hours**, which fits one session.

It does not re-download DroidCollection. Only the checkpoint and
`splits.parquet` are needed, so the 647 MB of raw parquets stay where they are.

If the checkpoint cannot be found, the notebook stops in cell 3 rather than
quietly starting a fresh 15-hour run.

---

## Moving a checkpoint between accounts

A notebook output is only visible to the account that produced it. To continue
a run somewhere else, the checkpoint has to travel as a Dataset.

**On the account that ran it:**

1. Open the notebook → **Output** tab.
2. Find `project/aicd/artifacts/branch_a_<tag>_ckpt.pt` and download it.
   It is roughly 1.8 GB: the model plus AdamW's two optimiser moments.
3. Also download `project/aicd/artifacts/data/splits.parquet`. Without it
   the run has no training data, and rebuilding it from scratch costs an hour.

**On the account that will continue it:**

4. Datasets → **New Dataset**, upload both files, title it `aicd-ckpt-<tag>`.
   Keep it **Private**; it is unpublished work.
5. Attach it to `4_resume_any.ipynb` alongside `aicd-code`.

The notebook finds the checkpoint by filename, so it does not matter which
input the file arrives in.

Do not rename the file. The tag is parsed out of `branch_a_<tag>_ckpt.pt`, and
that tag decides which config and which corpus get paired with it.

---

## Step 2 — Experiment 2: seed sweep

Same procedure with `kaggle_runs/2_seed_sweep.ipynb`. Trains two more models on
*exactly the original data*, differing only in the seed, and prints mean and
standard deviation across the three runs.

Report in Table III as mean ± sd. The S1-to-S5 gap is 0.66, so any ordinary
seed spread leaves the conclusion intact — this is insurance, not a gamble.

---

## When both are done

Download `results.zip` from each notebook's **Output** tab into
`kaggle_runs/results/`, then tell me and I will fold the numbers into the
paper, the tables and the verification chain.

The zip holds the JSON reports and the probability arrays. Model weights are
deliberately left behind: they are hundreds of megabytes and no number in the
paper needs them.

---

## Two bugs I fixed before writing this

Both would have silently wasted your GPU hours, and neither would have looked
like a failure.

**1. Runs overwrote each other.** Every training run wrote to
`branch_a_base_ckpt.pt`, `branch_a_base.json` and `proba_a_<slice>.npy`,
regardless of seed or config. Running seed 2 after seed 1 would have replaced
seed 1's results — and because the notebooks pass `--resume`, seed 2 would have
*resumed from seed 1's checkpoint*, producing one model trained twice rather
than two independent models. The output would have looked perfectly normal.

Fixed with a `--tag` flag. Each run now names its own artefacts, and the
notebooks pass `--tag matched`, `--tag seed1`, `--tag seed2`.

**2. The seed did nothing.** `project.seed` reached pandas sampling only.
Nothing called `torch.manual_seed`, so weight initialisation, dropout and batch
order came from torch's default entropy. A seed sweep would have produced
different numbers for reasons unrelated to the seed being swept. Fixed with
`seed_everything()`, covering `random`, `numpy` and `torch` including CUDA;
verified that the same seed reproduces and a different seed diverges.

---

## If something goes wrong

**"aicd/ not found under /kaggle/input"** — the dataset is not attached. Right
panel → Input → Add Input → Datasets → `aicd-code`.

**Cell 4 raises "Only NNN,NNN training rows"** — the extra shards were not
used. Delete the cached download and re-run:

```python
shutil.rmtree(WORK / "aicd" / "artifacts" / "data" / "hf", ignore_errors=True)
```

**Session killed mid-training** — use `4_resume_any.ipynb`. It reads every
checkpoint you attach, reports how far each run got, and continues the
unfinished ones. `--resume` restores model, optimiser, scheduler and scaler
from the last epoch checkpoint.

**Avoid losing a partial epoch.** The first matched-scale run was killed 43%
into epoch 2 and that GPU time was thrown away. `--max-hours` now stops cleanly
after the last epoch that fits in the budget; the resume notebook passes 9.5 h,
leaving room for evaluation inside the 12 h cap.

**Out of memory during training** — lower `batch_size` from 32 to 16 in
`aicd/configs/kaggle_matched.yaml`. Halving it roughly doubles step count but
each step is cheaper, so wall-clock cost is modest.

**Ran out of weekly GPU quota mid-run** — a committed run that is killed keeps
its checkpoint in the notebook's output. Attach that output as an input to a
fresh notebook next week and re-run the training cell with `--resume`.
