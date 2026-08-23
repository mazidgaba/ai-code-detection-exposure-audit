# Measured results

All numbers below are from an actual run on this machine, not from the papers.

**Setup.** 8-core CPU, no GPU, 16 GB RAM. Corpus: 180,001 rows sampled
stratified-by-label from all three `project-droid/DroidCollection` shards,
filtered to 144,939. Config `cpu.yaml` (8k TF-IDF features, 200 trees,
depth 6) — `base.yaml` holds the full-scale settings, which want a GPU.

## Corpus

| Stage | Rows | Removed |
|---|---|---|
| downloaded | 180,001 | — |
| min length | 179,982 | 19 (0.0%) |
| AST-parseable | 160,804 | 19,178 (10.7%) |
| length percentile (per language) | 144,986 | 15,818 (9.8%) |
| MinHash dedup @ 0.85 | 144,939 | 47 (0.0%) |

## Split matrix

Every slice carries all four classes, and zero `problem_id` overlap with
train. All 12 guarantee tests pass.

| Slice | Rows | human | machine | hybrid | adversarial |
|---|---:|---:|---:|---:|---:|
| train | 67,197 | 31,323 | 21,970 | 8,738 | 5,166 |
| val | 10,613 | 4,979 | 3,463 | 1,382 | 789 |
| s1 in-distribution | 10,618 | 4,963 | 3,421 | 1,415 | 819 |
| s2 unseen generator | 13,759 | 6,878 | 2,798 | 1,982 | 2,101 |
| s3 unseen language | 19,130 | 8,124 | 6,777 | 2,560 | 1,669 |
| s4 unseen domain | 22,093 | 12,291 | 3,163 | 3,762 | 2,877 |
| s5 compound | 1,526 | 836 | 305 | 254 | 131 |

One design note. Holding out the whole `general_deployed` domain — the literal
reading of the papers' setup — removed 83% of the corpus *and every adversarial
sample*, because adversarial code exists only in The Vault and StarCoder
sources. The holdout is therefore at source level (`THEVAULT_CLASS`,
`THEVAULT_INLINE`, `ARXIV`), which is a genuine structural shift (class-level
and inline snippets, plus research code) while keeping all four classes on
both sides.

## Branch B — TF-IDF + stylometry + AST -> XGBoost

| Slice | macro-F1 | weighted-F1 | accuracy | binary AUC | human FPR |
|---|---:|---:|---:|---:|---:|
| s1 in-distribution | 0.7227 | 0.8288 | 0.8235 | 0.9882 | 0.071 |
| s2 unseen generator | 0.6181 | 0.7342 | 0.7262 | 0.9919 | 0.071 |
| s3 unseen language | 0.3940 | 0.4886 | 0.4568 | 0.8497 | 0.339 |
| s4 unseen domain | 0.3139 | 0.3551 | 0.3396 | 0.7904 | 0.733 |
| s5 compound | 0.2091 | 0.2977 | 0.2654 | 0.7060 | 0.675 |

The degradation curve reproduces the published pattern, including the
magnitude at the hard end: **0.209 on compound shift against AICD Bench's
reported 0.21**. Domain shift hurts more than language shift, exactly as the
SemEval-2026 error analysis found.

The binary AUC column is the important caution. It stays at 0.99 in-distribution
while macro-F1 sits at 0.72 — human-vs-rest is easy, but telling machine from
hybrid from adversarial is not. A binary-only evaluation of this same model
would have looked excellent and told you almost nothing.

## Feature attribution (SHAP)

Top features on the dense set, mean |SHAP| over 1,500 samples:

| Rank | Feature | mean abs SHAP |
|---:|---|---:|
| 1 | `r_empty_lines` | 0.848 |
| 2 | `r_space_indent` | 0.389 |
| 3 | `r_indented_lines` | 0.345 |
| 4 | `char_entropy` | 0.262 |
| 5 | `mean_ident_len` | 0.259 |
| … | | |
| 19 | `ast_mean_depth` | 0.151 |

This reproduces CoDet-M4's finding directly: empty-line ratio dominates at
more than twice the next feature. Models space their code out consistently;
people do not.

## Calibration

| Slice | ECE raw | ECE calibrated | Brier | middle-band |
|---|---:|---:|---:|---:|
| s1 in-distribution | 0.0343 | **0.0087** | 0.041 | 0.178 |
| s2 unseen generator | 0.0475 | **0.0157** | 0.035 | 0.173 |
| s3 unseen language | 0.0919 | 0.0919 | 0.163 | 0.493 |
| s4 unseen domain | 0.3781 | 0.3564 | 0.352 | 0.206 |
| s5 compound | 0.2982 | 0.2982 | 0.325 | 0.534 |

Two published findings confirmed. Per-language isotonic regression cuts
in-distribution ECE by 4x, landing well inside the 0.05 promotion gate. And the
middle-band fraction never drops below 0.17, so this model does *not* have the
pathology Oedingen et al. found in their DNNs, which emitted almost nothing
between 0.1 and 0.9. Calibration collapses under domain shift (ECE 0.36) and
isotonic cannot repair it — the mapping itself was learned in-distribution.

## Decision policy

Thresholds fitted on val at a 1% target human false-positive rate:
`t_high = 0.938`, `t_low = 0.788`, abstain in between.

| Slice | coverage | abstain | human FPR | recall |
|---|---:|---:|---:|---:|
| s1 in-distribution | 0.933 | 0.067 | 0.014 | 0.925 |
| s2 unseen generator | 0.939 | 0.061 | 0.013 | 0.946 |
| s3 unseen language | 0.767 | 0.233 | 0.055 | 0.575 |
| s4 unseen domain | 0.841 | 0.159 | **0.577** | 0.940 |
| s5 compound | 0.734 | 0.266 | 0.250 | 0.638 |

**This is the most important table in the project.** A threshold fitted to
give a 1.4% false-positive rate in-distribution produces a **57.7%** false
positive rate on an unseen domain. The model flags more than half of all
human-written code as machine-generated, and does so confidently — it abstains
on only 16% of those inputs.

The fixed 0.15 abstention margin is not wide enough to absorb domain shift.
Two follow-ups worth doing: widen the band as a function of a detected
distribution shift, and gate the API on a domain classifier so unseen-domain
inputs abstain by default rather than being scored against a threshold that
does not apply to them.

## Formatting as an evasion

Existing trained model, scored against whitespace-normalized inputs (no
retraining) — the question "does running a formatter over AI code defeat the
deployed detector?"

| Slice | F1 raw | F1 formatted | delta | AUC raw | AUC formatted |
|---|---:|---:|---:|---:|---:|
| s1 in-distribution | 0.7219 | 0.4531 | **-0.269** | 0.987 | 0.873 |
| s2 unseen generator | 0.6304 | 0.4713 | -0.159 | 0.990 | 0.869 |
| s3 unseen language | 0.3752 | 0.3008 | -0.074 | 0.857 | 0.785 |
| s4 unseen domain | 0.3060 | 0.3174 | +0.011 | 0.785 | 0.854 |
| s5 compound | 0.2100 | 0.2507 | +0.041 | 0.707 | 0.733 |

A 27-point in-distribution drop, much larger than the ~4–8 points Oedingen
et al. reported. The two are not contradictory: they retrained on formatted
data so the model adapted, whereas this measures a deployed model meeting a
formatted input it was never trained for. Combined with SHAP putting
`r_empty_lines` first by 2x, the conclusion is that branch B leans heavily on
whitespace and a formatter is a cheap, effective evasion.

**Mitigation:** add formatter-normalized copies to the training set as
augmentation. That is the single highest-value change to make next.

## Line-level attribution

200 synthetic hybrids (human Python file, one function body deleted and
replaced), boundary detection with 40-line windows at 50% overlap:

- mean IoU 0.259, median 0.500
- 52% of predicted regions overlap the true region at all

Localizing a single-line insertion with 40-line windows is close to
impossible; this is a floor on the method rather than a defect. Narrower
windows would raise IoU at the cost of noisier per-window scores.

## Drift

PSI of the score distribution, in-distribution val as baseline vs. the
unseen-domain slice: **0.829 overall** (alarm threshold 0.25), with Java at
1.80 and C at 1.70. The monitor fires correctly on a genuinely shifted
distribution.

## Branches A and C

Both are implemented and verified to run; neither is trained at scale, because
this box has no GPU.

- **Branch A (ModernBERT + triplet):** verified on CPU with 64 training rows,
  loss 1.463 -> 1.247 over 2 epochs, evaluation path exercised on all five
  slices. Needs a GPU for a real run:
  `python -m aicd.models.modernbert_triplet --config base.yaml --resample`
- **Branch C (Fast-DetectGPT, Qwen2.5-Coder-0.5B):** verified on CPU, binary
  AUC 0.70 (val) and 0.74 (in-distribution) on 60-row samples — consistent
  with the 0.67–0.77 range published for Fast-DetectGPT on code. Full-slice
  scoring on CPU is impractical; it wants a GPU.
- **Stacker:** implemented, but it needs full-slice probabilities from A and C
  to be meaningful, so it has not been fitted on real data yet.

## What to do next, in order

1. **Formatter augmentation.** The 27-point evasion gap is the largest
   concrete weakness measured here, and it is cheap to close.
2. **Train branch A on a GPU** and fit the stacker. Expect in-distribution
   macro-F1 in the low 0.9s based on the published numbers.
3. **Make the abstention band shift-aware.** A 57.7% human FPR on unseen
   domains is the difference between a usable tool and a harmful one.
4. **Scale the corpus.** This run used 180k of the available 1.06M rows.
