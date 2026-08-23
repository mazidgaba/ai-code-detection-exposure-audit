# Held Out From What?

Research artifact for *Held Out From What? Distribution-Shift Benchmarks
Overstate the Robustness of AI-Generated Code Detectors.*

## What this is about

Benchmarks for AI-generated-code detection report out-of-distribution results by
withholding categories, such as unseen generator families, unseen programming
languages and unseen source domains, from the evaluation split. The withholding
is usually **benchmark-relative**: the category is absent from the benchmark's
own test partition, but nothing establishes that it was absent from the data the
**detector** was trained on.

When the two are separated, measured robustness changes substantially. This
repository holds the code, the measured results and the auditing tools behind
that finding.

## The protocol

The paper states it in three lines. A paper reporting out-of-distribution
results should:

1. record what the evaluated detector was trained on;
2. verify that the withheld categories are absent from that training data,
   rather than merely from the benchmark partition;
3. where the detector is external and its manifest is unavailable, restrict
   evaluation to a data shard the detector cannot have seen, and say that this
   restriction is in force.

Step 2 is implemented by `aicd/eval/exposure_audit.py`, which reads the category
columns of a named training split and reports the fraction of rows carrying each
withheld category. It needs the model card to say what the model was trained on,
and nothing else.

```bash
python -m aicd.eval.exposure_audit
```

## Repository layout

```
aicd/
  data/          corpus download, normalisation, filtering, splits, exposure arms
  models/        the trained detector, baselines, zero-shot scorer, transforms
  eval/          24 evaluation and audit modules
  eval/reports/  measured results as JSON, one file per experiment
  features/      stylometric, AST and TF-IDF feature builders
  scripts/       pipeline drivers and figure generation
  serve/         a FastAPI serving layer (not used by any paper experiment)
  tests/         72 tests
  configs/       17 YAML configurations, one per experimental arm
paper/
  review.py      audits every number in the manuscript against the reports
  validate.py    structural checks on the manuscript source
kaggle_runs/     notebook generators, the notebooks, and the run evidence
```

`aicd/` must stay at the repository root. Modules under `aicd/eval/` locate the
project with `Path(__file__).resolve().parents[2]`, so moving the package
changes what that resolves to.

## Requirements

Python 3.11 or later. Install with:

```bash
pip install -r requirements.txt
```

A GPU is required to retrain any model. It is **not** required to reproduce the
analyses: the stored probability arrays and reports are enough for most of them.

## Reproducing the analyses

Modules that read only the stored results run directly:

```bash
python -m aicd.eval.e5_ablation        # architecture ablation against seed spread
python -m aicd.eval.duplication_audit --self-test
python paper/review.py                 # re-audit the manuscript's numbers
python -m pytest aicd/tests -q         # 72 tests
```

`aicd/eval/reports/*.json` holds the measured result of each experiment.
`kaggle_runs/results/` holds the per-row probability arrays and the kernel logs
from the GPU runs.

Two arms of the architecture ablation, `proj256` and `triplet02`, ran on Kaggle
accounts that were later rotated away, and only their kernel logs were retained.
Their macro-F1 figures are complete, and `aicd/eval/e5_ablation.py` reads them
from those logs; the field `"provenance": "kernel log"` in
`aicd/eval/reports/e5_ablation.json` records which arms those are. Their
probability arrays are not in this repository.

## Data

**No corpus is redistributed here.** Each is obtained from its own source under
its own terms.

| Dataset | Source | Used for |
|---|---|---|
| DroidCollection | `project-droid/DroidCollection` on Hugging Face | training and the five evaluation conditions |
| CodeMirage | `HanxiGuo/CodeMirage` on Hugging Face | external validation |
| AIGCodeSet | `basakdemirok/AIGCodeSet` on Hugging Face | external validation |
| DroidDetect-Base | `project-droid/DroidDetect-Base` on Hugging Face | the published detector under audit |

Check each source's licence before redistributing anything derived from it.
AIGCodeSet states CDLA-Permissive-2.0 in its dataset card; the others state their
terms on their own pages.

Downloads land under `aicd/artifacts/`, which is not tracked here:

```
aicd/artifacts/
  data/
    hf/                 raw DroidCollection parquet shards
    splits.parquet      the five conditions after filtering
    independent/        CodeMirage and AIGCodeSet
  droiddetect/          DroidDetect-Base weights
```

To build the corpus from scratch:

```bash
python -m aicd.data.download  --config base.yaml --train-shards 1
python -m aicd.data.normalize --config base.yaml
python -m aicd.data.filter    --config base.yaml
python -m aicd.data.splits    --config base.yaml
```

## Limitations

The exposure audit settles whether the withheld categories are present in the
training split a model card names. It cannot settle whether the publisher's own
filtering removed them afterwards, because that filter is not released. The
distinction is recorded in the audit's output.

`aicd/serve/` is a serving layer that no experiment in the paper exercises. It
is included because it is working code, not because any result depends on it.

## Authorship

The paper is by Gulam Mazid (National Institute of Technology, Warangal), Saquib
Warsi (Indian Council of Medical Research) and Md Jamaluddin (Aligarh Muslim
University).

Much of the code in this repository was written with AI assistance. The
experimental design, the results and the conclusions are the authors'.

## Citation

The paper is not yet published. Cite this repository by its archived version
until it is.

## Licence

MIT. See [LICENSE](LICENSE).
