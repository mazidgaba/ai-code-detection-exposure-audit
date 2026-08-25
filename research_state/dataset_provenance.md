# Dataset provenance

## Source

**DroidCollection** (`project-droid/DroidCollection`, Hugging Face). Raw shards
as pulled, with adversarial counts, from `shard_provenance.json`:

| Shard | Rows | Adversarial |
|---|---|---|
| train-00000-of-00003 | 282,200 | 0 |
| train-00001-of-00003 | 282,199 | 0 |
| train-00002-of-00003 | 282,199 | 129,115 |
| dev-00000-of-00001 | 105,824 | 16,139 |
| test-00000-of-00001 | 105,826 | 16,140 |

## Two builds, never interchanged

| | GPU build | CPU build |
|---|---|---|
| Training rows | 196,854 | smaller |
| Evaluation rows | 220,670 | 77,739 |
| Used for | every headline result | the earlier contamination measurement only |

`verification/review.py` loads GPU-build figures from `eval/reports/kaggle/` and
CPU-build figures from `eval/reports/`, and the manuscript states which build
each table comes from. Interchanging them is the failure mode most likely to
produce a plausible falsehood here, which is why the auditor separates them by
directory rather than by convention.

Condition sizes on the GPU build: S1 31,083; S2 41,061; S3 56,631; S4 56,731;
S5 4,168; validation 31,069; training 196,854; unused 48.

## A structural property that matters

**The training shard contributes no adversarial rows to this corpus.** The
adversarial class survives filtering almost entirely from dev and test: of
15,959 adversarial rows in the build, 7,980 come from dev and 7,979 from test,
with a survival rate of 0.4944.

This is not a curiosity. It means any comparison between train-shard rows and
held-out rows must average over the classes present in both, because a
four-class macro-F1 charges the train arm for a class it cannot contain. Doing
it the naive way returns a mean gap of **-0.115**, in the wrong direction, for a
reason that has nothing to do with memorisation. The class-matched comparison
returns **+0.1106**. See `contamination_gpu.json` and the regression test in
`aicd/tests/test_resample.py`.

## The exposure-arm build

E1's twin arms use a separate build (`splits_arms.parquet`). S1 and S5 are
**protected**: no row of either is moved into training in any arm. S2, S3 and S4
are **donors**: the exposed arm draws its additional training rows from them, so
their evaluation sets are smaller in this build than in the standard one:

| Condition | Arm build | Standard |
|---|---|---|
| S2 unseen generator | 30,344 | 41,058 |
| S3 unseen language | 26,936 | 56,626 |
| S4 unseen source | 27,100 | 56,728 |

Paired comparisons between the arms are therefore valid only on S1 and S5.
`aicd/eval/paired.py` verifies row counts before pairing and skips rather than
truncating.

## External corpora

| Corpus | Rows used | Overlap with ours | Notes |
|---|---|---|---|
| AIGCodeSet | 7,202 after filtering | 3 of 7,471 (0.04%) | 266 internal duplicates removed; binary, mapped to three of our four classes |
| CodeMirage | 62,964 | 1 of 62,964 | HTML and Ruby absent from DroidCollection entirely; composition 4.8% human / 47.6% machine / 47.6% adversarial |

None of these is redistributed. Each is obtained from its own source under its
own terms, as the artifact README documents.

## Label vector

`aicd/artifacts/keys/eval_row_keys.parquet` holds label, condition and shard for
all 220,670 GPU-build evaluation rows. Recovered in E10 from the deterministic
corpus build. Validated on entry to every module that uses it: it reproduces
each stored report's macro-F1 to 1e-9 across all five conditions.

Earlier runs saved probability arrays without this vector, which is why paired
comparisons were impossible until E10 and why aggregate statistics survived the
gap while row-level ones did not.
