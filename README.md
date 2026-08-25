# Held Out From What?

Code and measured results for *Held Out From What? Evaluating Distribution
Shift Relative to Training Exposure in AI-Generated Code Detection.*

## The problem

Benchmarks for AI-generated-code detection report out-of-distribution results by
withholding a category, such as a generator family, a programming language or
a code source, from the evaluation split. That withholding is nearly always
**benchmark-relative**: the category is absent from the benchmark's own test
partition, and nothing establishes that it was absent from the data the
**detector** was trained on. Where a benchmark and a detector are built from the
same collection, the second usually does not follow from the first.

Separating the two changes what the numbers mean. This repository holds the code
that separates them and the results that follow.

## The protocol

Three lines, and step 2 is the one that is usually skipped:

1. record what the evaluated detector was trained on;
2. verify that the withheld categories are absent from *that training data*,
   not merely from the benchmark partition;
3. where the detector is external and its manifest is unavailable, restrict
   evaluation to a shard the detector cannot have seen, and say so.

Step 2 needs the model card to state the training split, and nothing else:

```bash
python -m aicd.eval.exposure_audit
```

## Layout

```
aicd/
  data/        corpus download, filtering, splits, exposure arms
  models/      the trained detector, the classical baseline, the zero-shot scorer
  features/    stylometric and AST features for the classical baseline
  eval/        the analyses; one module per question
  eval/reports/  the measured result of each, as JSON
  scripts/     pipeline drivers and figure generation
  tests/       assertions that gate the protocol
  configs/     one YAML per experimental arm
kaggle_runs/   the notebooks that ran on GPU, with their per-row probability
               arrays and kernel logs
research_state/  the ledgers described below
paper/         the audit that checks the manuscript against these reports
docs/          measured results, and the retraining runbook
```

## Reproducing

Most analyses read stored arrays and run in seconds on a laptop. Nothing here
needs a GPU unless you are retraining.

These run on the stored arrays alone. Each rewrites its own report, so you can
delete `aicd/eval/reports/<name>.json` and watch it come back with the same
numbers.

```bash
pip install -r requirements.txt

python -m aicd.eval.seed_twin             # the twin effect across three seeds
python -m aicd.eval.matched_axis          # the two axes on comparable rows
python -m aicd.eval.language_clean        # the arm with no untrained language
python -m aicd.eval.published_shift       # the published detector on rows
                                          # outside its own training shard
python -m aicd.eval.independent_taxonomy  # external collapse, with and without
                                          # the class we mapped ourselves
python -m aicd.eval.variance_components   # design against seed variation
python -m pytest aicd/tests -q
```

Two need the corpus, because they count categories in it rather than reading a
stored result. Both stop with a message saying so if it is absent:

```bash
python -m aicd.data.download              # pulls DroidCollection from HF first
python -m aicd.eval.exposure_audit        # categories present in a named split
python -m aicd.eval.aicdbench_audit       # the same, but using a third party's
                                          # withheld categories rather than ours
```

`aicd/eval/reports/*.json` holds the measured result of each experiment.
`kaggle_runs/results/` holds the probability arrays and kernel logs from the
runs that needed a GPU.

Two arms of the architecture ablation, `proj256` and `triplet02`, ran on
accounts that were later rotated away and only their kernel logs survive. Their
macro-F1 figures are complete and `aicd/eval/e5_ablation.py` reads them from
those logs; the `"provenance": "kernel log"` field in
`aicd/eval/reports/e5_ablation.json` marks which arms those are. Their
probability arrays are not here.

## Checking a number without running anything

`research_state/numbers_ledger.csv` lists every figure the manuscript quotes
next to the file it came from and whether the two agree. `claims_ledger.csv`
lists what each claim rests on, including the two results that did not come out
the way the argument predicted: the exposure gradient did not reproduce on the
second external corpus, and the transformation battery was never run across the
detector panel, so no claim is made that it generalises.

`paper/review.py` regenerates that ledger by parsing the manuscript and checking
each table cell against its source. The manuscript is not distributed here,
because the paper is unpublished, so the script says so rather than failing. It
is included because it documents what was checked.

## Data

**No corpus is redistributed.** Each is obtained from its own source under its
own terms:

- DroidCollection, from Hugging Face, by `aicd/data/download.py`
- AIGCodeSet and CodeMirage, by their own loaders under `aicd/eval/`

Model weights are not included either. `aicd/data/` rebuilds the corpus and
splits from the published sources.

Two builds exist and their filtering rates differ, so it is worth being explicit
about which is which. The paper's headline figures come from the larger build:
493,850 rows surviving normalisation, of which 417,645 remain after filtering,
with the parse check removing 6.1% and the length filter 9.9%. The log shipped
here, `aicd/eval/reports/filter_log.txt`, is the smaller build used for CPU
development: 180,001 rows in, 144,939 out, parse removing 10.7% and length 9.8%.
The rates differ because the builds draw different numbers of shards, not
because the pipeline changed. Compare a rebuild against whichever of the two it
reproduces.

## Limitations

The detector we train is one architecture on one collection, and the
hyperparameter space around it is unexplored: the values used are the published
recipe's where it states them and library defaults elsewhere. Results attributed
to DroidDetect-Base come from our reconstruction of it around the published
weights, not from an inference path supplied by its authors; the same procedure
applied to DroidDetect-Large produces a degenerate model, which is reported
rather than set aside.

## Authors

Gulam Mazid (National Institute of Technology, Warangal), Saquib Warsi (Indian
Council of Medical Research), Md Jamaluddin (Aligarh Muslim University) and
Afrah Fathima (Maulana Azad National Urdu University, Hyderabad).

## How this was built

Parts of the code here were written with AI assistance, as is now common. The
experimental design, the choice of controls, the interpretation of the results
and the conclusions are the authors'. Every number in the paper is checked
against the report file that produced it by `paper/review.py`, and the checking
code is itself mutation-tested, which is the reason to trust the figures rather
than the provenance of any particular line.

## Citation

The paper is not yet published. Cite this repository until it is.

## Licence

MIT. See [LICENSE](LICENSE).
