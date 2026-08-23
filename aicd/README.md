# AI-generated code detection

Four-class provenance classifier for source code: **human**, **machine**,
**hybrid** (human code completed or rewritten by an LLM), and **adversarial**
(an LLM prompted or tuned to imitate human style).

Built from the findings in four papers plus three current benchmarks. The
design constraints below are consequences of published results, not
preferences — breaking any of them invalidates the evaluation.

## Why it is shaped this way

| Constraint | Consequence of |
|---|---|
| Four classes, never binary | CoDet-M4: hybrid F1 collapses 86 → 39 under binary training |
| Split by `problem_id`, never by row | Oedingen et al.: a random split inflates every metric ~4 points |
| Report five OOD slices, never one number | AICD Bench: best detector 44.99 macro-F1 vs 45.73 random under shift |
| Keep the TF-IDF/XGBoost branch permanently | AICD Bench: classical models beat all six neural encoders under shift |
| Train on adversarial data | Droid: adversarial recall 0.92 with it, ~0.10 without |
| Emit calibrated probabilities + abstention | Oedingen et al.: DNNs emitted only ~0.0 and ~1.0 — confidently wrong |

## Layout

```
aicd/
  config.py            YAML config with `extends` support
  configs/             base.yaml (full run), smoke.yaml (tiny)
  data/
    download.py        pull DroidCollection parquet from HuggingFace
    normalize.py       canonical schema + problem_id grouping
    filter.py          parse check, length percentile, MinHash dedup
    splits.py          THE OOD SPLIT MATRIX -- five slices
    generate.py        build your own machine/hybrid/adversarial samples
    stats.py           corpus composition, thin-cell warnings
  features/
    stylometry.py      33 features (Oedingen's 7 + CoDet-M4 structural)
    ast_feats.py       tree-sitter depth + per-node-type densities
    tfidf.py           character n-grams, 2-5
    build.py           assemble + per-language feature selection
  models/
    xgb.py             BRANCH B -- the robustness anchor
    modernbert_triplet.py  BRANCH A -- CE + 0.1 x batch-hard triplet
    fastdetect.py      BRANCH C -- zero-shot curvature
    stacker.py         fusion, fitted on val only
    shap_report.py     which human-readable features carry signal
    formatter_ablation.py  how much of the signal is just formatting
  eval/
    metrics.py         per-slice macro-F1, AUC, human-FPR
    calibration.py     ECE, reliability, overconfidence alarm
    drift.py           PSI monitoring
  serve/
    runtime.py         load whichever branches exist
    policy.py          thresholds from target human-FPR, abstain band
    attribution.py     line-level attribution for hybrid code
    api.py             FastAPI
  scripts/
    smoke.py           end-to-end test on synthetic rows, ~10s
    run_pipeline.py    stage runner
```

## Run it

```bash
# 10-second wiring check
python -m aicd.scripts.smoke

# full pipeline (CPU-friendly stages)
python -m aicd.scripts.run_pipeline

# with the neural branches (wants a GPU)
python -m aicd.scripts.run_pipeline --with-neural

# serve
python -m uvicorn aicd.serve.api:app --port 8000
```

Branch A on CPU is only useful for verifying the training loop runs:

```bash
python -m aicd.models.modernbert_triplet --cpu --max-train 200
```

For a real run, use a GPU box and drop `--cpu --max-train`.

## The five slices

| Slice | What is held out | Expect |
|---|---|---|
| `s1_in_distribution` | nothing | 0.94–0.99 — the vanity metric |
| `s2_unseen_generator` | two model families | 0.85–0.93 |
| `s3_unseen_language` | Go, JavaScript | 0.75–0.89 |
| `s4_unseen_domain` | The Vault class/inline, arXiv | 0.45–0.62 — the real bottleneck |
| `s5_compound` | unseen language *and* source | 0.20–0.45 — near random |

A high abstention rate on `s4` and `s5` is correct behaviour, not a bug to
tune away.

## Reading the output

`/analyze` returns four class probabilities, a decision (which may be
`abstain`), the branches that contributed, and a `confidence_note`. That note
is required on every response. A positive result is **provenance evidence, not
proof of authorship** — corroborate with commit history, authoring telemetry,
or a code walkthrough before anyone acts on it.

Twenty experienced programmers scored 48.7% on this task by eye, worse than a
coin flip, so there is no human fallback to appeal to when a reviewer disagrees
with the model. That cuts both ways: the model is not obviously right either.
