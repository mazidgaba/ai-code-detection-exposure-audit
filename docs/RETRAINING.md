# Retraining runbook

A detector for AI-generated code goes stale in months. New model families ship
with different output distributions, and nothing in an incoming request tells
you that has happened — the model keeps returning confident probabilities
against a distribution it has never seen.

Retrain on a fixed quarterly cadence, and out of cycle whenever drift fires.

## Triggers

| Trigger | Signal | Action |
|---|---|---|
| Scheduled | Every 90 days | Full refresh |
| Drift alarm | PSI > 0.25 overall, or > 0.25 for one language two weeks running | Full refresh |
| New major generator | A widely-adopted model family ships | Add generator, partial refresh |
| Abstention climb | Abstain rate up >10 points on production traffic | Investigate before retraining |

Check drift with:

```bash
python -m aicd.eval.drift --check-slice s1_in_distribution
```

## Full refresh

1. **Pull new upstream data.** `project-droid/DroidCollection` and
   `AICD-bench/AICD-Bench` both get updated. Re-run `data/download.py`.

2. **Generate against current models.** This is the step that keeps the
   detector current. Aim for a few thousand samples per new family, across all
   four modes:

   ```bash
   python -m aicd.data.generate --mode machine     --n 2000 --model <new-model>
   python -m aicd.data.generate --mode hybrid      --n 1000 --model <new-model>
   python -m aicd.data.generate --mode adversarial --n 1000 --model <new-model>
   python -m aicd.data.generate --mode mutate      --n 1000
   ```

   Do not skip `adversarial`. Detectors trained without it score around 0.10
   recall on evasion attempts; with it, around 0.92. It is the highest-value
   data in the corpus per sample.

3. **Re-run the quality gate.** Generated rows go through exactly the same
   pipeline as the corpus — no exceptions, or you introduce a length or
   formatting artifact that the model will happily learn.

   ```bash
   python -m aicd.data.filter && python -m aicd.data.splits && pytest aicd/tests/ -q
   ```

   The split tests are the gate. If any fail, stop: every metric downstream is
   meaningless until they pass.

4. **Rotate the holdouts.** Change `holdout_families`, `holdout_languages`, and
   `holdout_sources` in `configs/base.yaml` each cycle. Keeping the same
   holdouts across refreshes turns the OOD slices into a second training set
   that you tune against by hand.

5. **Retrain and evaluate.**

   ```bash
   python -m aicd.scripts.run_pipeline --from features --with-neural
   ```

6. **Refit thresholds and calibration.** These are not transferable across
   model versions. A stale threshold with a fresh model is how a system starts
   making confident false accusations.

## Promotion criteria

Promote a new model only if **all** hold, compared against the model in
production:

- In-distribution macro-F1 within 1 point or better.
- No OOD slice regresses by more than 3 points of macro-F1.
- Human false-positive rate at the operating threshold is at or below the
  configured target (default 1%).
- Expected calibration error below 0.05 on the in-distribution slice.
- Middle-band fraction at least 0.05 — the model must produce probabilities
  between 0.1 and 0.9. A model that only ever emits near-0 and near-1 is not
  calibrated, and its confidence is not usable.
- The compound-shift slice still abstains heavily. If abstention there drops
  sharply, treat it as a red flag, not an improvement: the published ceiling
  for that setting is near random, so a confident model is probably reading a
  corpus artifact.

## What not to do

- Do not tune thresholds against an OOD slice to make the numbers look better.
  Those slices are the only honest estimate you have of production behaviour.
- Do not drop the abstain band because users find it unsatisfying. It is the
  difference between a tool that reports evidence and one that fabricates
  certainty.
- Do not remove branch B because branch A scores higher in-distribution.
  Branch B is what holds up under shift.
