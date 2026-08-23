# The simple way to continue

No downloads. No API token. No moving files between accounts.

Each account continues its own work. That is all.

---

# Account 1 — the one that ran "Matched-scale"

It finished 2 of 3 epochs before the 12-hour limit stopped it. The unfinished
model is saved inside that notebook's output, so you just point a new notebook
at it.

### Step 1. Update the code (2 minutes)

1. Go to **kaggle.com/datasets**
2. Click your **aicd-code** dataset
3. Click **New Version**
4. Upload `kaggle/aicd-code.zip` from your computer
5. Click **Create**

Do not skip this. The old code does not have the resume features and the run
will fail after a few minutes.

### Step 2. Make the new notebook (3 minutes)

1. Go to **kaggle.com/code**
2. Click **New Notebook**
3. Top menu: **File → Import Notebook**
4. Upload `kaggle_runs/4_resume_any.ipynb` from your computer

### Step 3. Set it up (2 minutes)

Look at the panel on the right side of the screen.

1. **Accelerator** → choose **GPU T4 x2**
2. **Internet** → turn **On**
3. Click **+ Add Input**. A window opens.
   - Search `aicd-code`, click **Add**
   - Now click the **Notebooks** tab at the top of that window
   - Find **Matched-scale**, click **Add**
   - Close the window

You should now see two things listed under Input.

### Step 4. Start it

1. Click **Save Version** (top right)
2. Choose **Save & Run All (Commit)**
3. Click **Save**
4. Close the browser tab

It now runs on Kaggle's computers. Your laptop can be off. It takes about
**7 hours**.

---

# Account 2 — the one that ran "seed_sweep"

It **did** train, despite the Output tab looking empty. The log shows it
reached epoch 1, so at least one epoch finished and a checkpoint was saved.
The Output tab only lists files at the top level, which is why `aicd` looked
empty when the checkpoint was sitting in a subfolder.

So it continues exactly like Account 1.

### Step 1. Update the code

Same as Account 1, Step 1. Upload `kaggle/aicd-code.zip` as a **New Version**
of that account's `aicd-code` dataset.

### Step 2. Make the notebook

1. **New Notebook** → **File → Import Notebook**
2. Upload `kaggle_runs/4_resume_any.ipynb`   (the same one, not 2_seed_sweep)

### Step 3. Set it up

1. **Accelerator** → **GPU T4 x2**
2. **Internet** → **On**
3. **+ Add Input**:
   - search `aicd-code`, click **Add**
   - click the **Notebooks** tab, find **seed_sweep**, click **Add**

### Step 4. Start it

**Save Version → Save & Run All (Commit) → Save**, then close the tab.

### How long

Cell 3 prints what it found before anything trains. Either:

| Checkpoint says | Remaining | Time |
|---|---|---|
| 1 epoch done | 2 epochs + evaluation | about 5.8 h |
| 2 epochs done | 1 epoch + evaluation | about 3.4 h |

A seed run is smaller than matched-scale: 6,151 steps per epoch against
12,332, so 2.35 h per epoch rather than 4.5 h.

---

# If you see "unrecognized arguments"

```
modernbert_triplet.py: error: unrecognized arguments: --max-hours 9.5
```

The `aicd-code` dataset attached to the notebook is an older version than the
zip on your computer. Two things have to happen, and the second is the one
people miss.

### 1. Upload the current zip

kaggle.com/datasets → your **aicd-code** → **New Version** → upload
`kaggle/aicd-code.zip` → **Create**. Wait for it to finish processing.

### 2. Point the notebook at the new version

Uploading a new version does **not** move existing notebooks onto it. Kaggle
pins each notebook to the version that was current when you attached it.

In the notebook, in the right-hand **Input** panel, find `aicd-code`. If it
offers a version number or an update link, switch it to the newest. If you
cannot find that control, the reliable fix is to remove the input and add it
again:

1. Hover over `aicd-code` in the Input panel, click the **x** to remove it
2. **+ Add Input** → search `aicd-code` → **Add**

Re-adding always attaches the latest version.

Then **Save Version → Save & Run All (Commit)** again.

The notebook now checks which flags the attached code supports and adapts, so
an older dataset will no longer kill the run. It will print a note instead.
But updating is still worth doing: without `--max-hours` a long run can be
killed mid-epoch and lose that epoch.

---

# Before you start: check you have GPU hours

Kaggle gives 30 GPU hours per week and resets them weekly.

With any notebook open, look at the **top of the right-hand panel**. It shows
how many hours you have left and when they reset.

- Enough hours → start now.
- Nearly zero → wait for the reset date shown there. Starting anyway just
  makes the notebook sit and wait, then get cancelled.

Both accounts ran for the full 12 hours, so both will have used 12 of their
30. That leaves roughly 18 hours each, which is more than enough: the two
resumes need about 7 h and 6 h.

---

# Two rules that matter

**Always use "Save & Run All (Commit)".** Never the plain Run button. Save &
Run All keeps going on Kaggle's servers after you close the browser. The plain
Run button stops the moment your browser disconnects or your laptop sleeps.

**Only one notebook at a time per account.** Two GPU notebooks on the same
account do not run side by side. The second one waits, and then gets cancelled
without ever training. That is most likely what happened to your seed sweep.

---

# What you should see

Open the notebook's **Logs** tab any time to check on it.

On Account 1, early in the run, you will see:

```
tag           epochs done  remaining  est. hours
--------------------------------------------------
matched             2 of 3          1         4.9
```

That means it found your saved work and is finishing the last epoch. Good.

If instead it says `No branch_a_*_ckpt.pt in any attached input`, you missed
the Notebooks tab in Step 3. Go back, add the Matched-scale notebook, and start
again. Nothing is lost.

---

# When it finishes

Open the notebook → **Output** tab → download **results.zip**.

Put it in `kaggle_runs/results/` on your computer and tell me. I will put the
numbers into the paper.
