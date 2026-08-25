# Reproduction status

Checked 2026-08-24, on this machine, from the released tree.

## Automated verification

| Check | Result |
|---|---|
| Test suite | 91 passed in 17.72s |
| Manuscript numbers traced to source files | 343 values traced, all `ok` |
| Structural validation (`paper/validate.py`) | PASS |
| PDF compiles | clean, 18 pages |

`research_state/numbers_ledger.csv` records every trace: label, source value,
manuscript value, tolerance, status. Seventeen tables covered.

## What reproduces without a GPU

These read stored probabilities or reports and complete in seconds to minutes:

```
python -m aicd.eval.e5_ablation          # ablation against the seed spread
python -m aicd.eval.e3_panel             # (report already built)
python -m aicd.eval.augment_recover      # E8 from its kernel log
python -m aicd.eval.transform_recover    # E6 from its kernel log
python -m aicd.eval.comparisons          # seed effect size, Holm correction
python -m aicd.eval.intervals            # per-class CIs from confusion matrices
python -m aicd.eval.paired               # twin paired differences
python -m aicd.eval.duplication_audit --self-test
python paper/review.py                   # re-audit and re-emit the ledger
```

## What needs a GPU

Retraining any arm, and any scoring of a detector over the corpus. Roughly 8 to
9 hours per trained arm on a T4; inference over the evaluation set is 30 to 60
minutes depending on the model.

## What cannot be reproduced from the release

- **Per-row probabilities for five runs.** `proj256`, `triplet02`, E6, E8 and
  the E3 panel ran on compute accounts later rotated away, and only their kernel
  logs were retained. Their per-condition tables are complete and parsed by
  recovery modules that cross-check the parse against the log's own summary
  line. Each report carries `"provenance": "kernel log"`.
- **The corpora themselves**, by licence. Each is obtained from its own source.

## Known reproduction hazards

- Two corpus builds exist and must not be interchanged; the auditor separates
  them by directory.
- `config.ROOT` is the package directory, while `aicd/eval/` modules define
  `ROOT` as the project directory. Joining a package-relative path onto the
  former writes to `aicd/aicd/...`. This cost one run's report; a test now
  pins it.
- Kaggle kernel outputs include multi-gigabyte checkpoints that break a plain
  download. Fetch `results.zip` by file pattern instead.
