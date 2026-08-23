# Step by step, from where you are now

State: the matched-scale run was killed at the 12 h cap with 2 of 3 epochs
done, and its checkpoint survived in a 4.64 GB output. The seed sweep produced
an empty output, so it never wrote a checkpoint and has to start over. That is
the lower-priority run, so leave it until matched-scale is finished.

Work through this in order. Steps 1 and 2 are quick and protect everything
else, so do not skip them.

---

## Step 1 — Pull the checkpoints to your own disk (~20 min)

Do this **before anything else**. Right now the only copy of 12 hours of GPU
work lives inside a Kaggle notebook output.

`fetch_outputs.py` lists a notebook's output and downloads only what a resume
needs, roughly 2 GB of the 4.64 GB rather than all of it.

**One-time setup.** Open <https://www.kaggle.com/settings> → Account → API →
**Create New Token**. Kaggle now shows a `KGAT_...` string rather than
downloading a `kaggle.json`, and it is shown **once**.

Save it in PowerShell, replacing the placeholder with your own token:

```powershell
$t = "KGAT_paste_yours_here"
New-Item -ItemType Directory -Force "$HOME/.kaggle" | Out-Null
[IO.File]::WriteAllText("$HOME/.kaggle/access_token", $t)
```

`WriteAllText` matters: `Set-Content` and `>` append a newline, and some
clients send the trailing byte and get a 401 that looks like a bad token.

Setting `KAGGLE_API_TOKEN` in the environment works too. A legacy
`kaggle.json` with username and key is still accepted if you have one.

Never paste a token into a screenshot, a chat, or a commit. Anyone holding it
can act as you on Kaggle. If one leaks, open the same page and create a new
token; issuing a new one invalidates the old.

Use the token of the account that **owns** the notebook. A private notebook's
output is visible only to its owner, so repeat this with the second account's
token for the second notebook.

**Look before you download:**

```
python kaggle_runs/fetch_outputs.py --list gulammazid786/matched-scale
```

That prints every output file with its size and marks the ones it would take.
If no `branch_a_*_ckpt.pt` appears, that run never finished an epoch and has
to start over rather than resume.

**Then fetch:**

```
python kaggle_runs/fetch_outputs.py gulammazid786/matched-scale
```

Files land in `kaggle_runs/results/<slug>/`, and the script prints exactly
which two to upload in step 4. Re-running skips anything already downloaded,
so an interrupted pull is safe to repeat.

Take the slug from the notebook URL: `kaggle.com/code/gulammazid786/seed-sweep`
gives `gulammazid786/seed-sweep`.

---

## Step 2 — Update the code dataset on **both** accounts (~5 min each)

The notebooks now use three flags that did not exist yesterday: `--tag`,
`--max-hours` and working seeding. An old code dataset will fail with an
unrecognised-argument error a few minutes into the run.

On each account:

1. Datasets → open your `aicd-code` dataset.
2. **New Version** → upload `kaggle/aicd-code.zip` (118 KB) → **Create**.

If an account has no `aicd-code` dataset yet: Datasets → **New Dataset** →
upload the zip → title it exactly `aicd-code` → **Create**.

---

## Step 3 — Pick the account with quota

Open any notebook on each account and read the top of the right-hand panel. It
shows GPU hours left this week and the reset date.

Use whichever has the most. If both are near zero, everything below waits for
the reset; there is no way around it.

---

## Step 4 — Put the checkpoint where the quota is

**Skip this if the account with quota is the one that already holds the
checkpoint** — its own notebook output can be attached directly in step 5.

Otherwise, on the account you are going to run on:

1. Datasets → **New Dataset**.
2. Upload the two files from step 1: `branch_a_matched_ckpt.pt` and
   `splits.parquet`.
3. Title it `aicd-ckpt-matched`. Set visibility **Private**.
4. **Create**, and wait for it to finish processing.

Do not rename the checkpoint file. The notebook parses the run's tag out of
`branch_a_<tag>_ckpt.pt`, and that tag selects the config and corpus.

---

## Step 5 — Run the resume notebook

1. Code → **New Notebook**.
2. **File → Import Notebook** → upload `kaggle_runs/4_resume_any.ipynb`.
3. Right-hand panel:
   - **Accelerator**: `GPU T4 x2`
   - **Internet**: `On`
   - **Persistence**: `Files only`
   - **Input → Add Input**, and add **both**:
     - Datasets → `aicd-code`
     - the checkpoint source: either **Your Work → Notebooks →** the killed
       run, or Datasets → `aicd-ckpt-matched` from step 4
4. **Save Version** → **Save & Run All (Commit)** → **Save**.
5. Close the tab.

Use Save & Run All, not the Run button. An interactive session dies with your
browser; a committed run continues on Kaggle's servers.

---

## Step 6 — Read cell 3 before walking away

Cell 3 prints exactly what it found and what remains, before any training
starts:

```
tag           epochs done  remaining  est. hours
--------------------------------------------------
matched             2 of 3          1         4.9
```

If it says `No branch_a_*_ckpt.pt in any attached input`, the checkpoint is not
attached. Fix the input and re-run; nothing has been wasted.

Expect roughly **7 hours**: one epoch at 4.9 h plus about 2 h of evaluation.

---

## Step 7 — If it stops early, that is correct behaviour

With `--max-hours 9.5` the run now stops cleanly after the last epoch that
fits, instead of being killed mid-epoch:

```
stopping after epoch 1: 9.80 h used, next epoch needs about 4.90 h,
budget is 9.50 h. re-run with --resume to continue from epoch 2.
```

That is the fix for what happened last time, when 2.1 h of GPU was thrown away
on a partial epoch. Just run the same notebook again next session.

---

## Step 8 — Repeat for the seed runs

The seed sweep left an empty output, so there is no checkpoint to resume from
and it starts over with `2_seed_sweep.ipynb` rather than `4_resume_any.ipynb`.

Before re-running it, confirm the first attempt was not simply starved of GPU:
two committed GPU runs on one account queue rather than run side by side, so a
seed sweep started while matched-scale was still going would wait and then be
cancelled without training. Run one at a time.

It needs 8.6 h per seed. If quota is short, one extra seed still lets you state
the spread honestly, and two seeds plus a finished matched-scale control beats
three seeds and no control.

Once a seed checkpoint exists, `4_resume_any.ipynb` picks it up like any other.
Cell 4 has `ONLY = None`, which runs everything it finds; set it to `"seed1"`
to do one at a time. Matched-scale is always attempted first.

---

## Step 9 — Send me the results

Same tool as step 1:

```
python kaggle_runs/fetch_outputs.py <user>/<slug>
```

It takes the JSON reports and the probability arrays, which is everything the
analysis needs, and leaves the model weights behind. Then tell me, and I will
fold the numbers into the paper, the tables, and the verification chain.

---

## Quick reference

| Situation | Do this |
|---|---|
| Checkpoint not found | Attach the notebook output or the checkpoint dataset |
| `unrecognized arguments: --tag` | Code dataset is stale, redo step 2 |
| `No GPU` error | Quota exhausted, or accelerator not set to GPU T4 x2 |
| Cell 2 hangs | Internet is Off in the right-hand panel |
| `401` from fetch_outputs.py | Token expired, or a newline crept into access_token |
| `403` from fetch_outputs.py | Token belongs to the other account; use the owner's |
| Killed at 12 h again | Lower `BUDGET_H` in cell 4 from 9.5 to about 8 |
| Everything queued | Only one GPU session at a time per account |
