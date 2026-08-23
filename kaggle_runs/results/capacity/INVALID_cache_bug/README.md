# Invalid: do not use these numbers

These `droiddetect_baseline_large*.json` reports are Base's results under
Large's name. The scoring cache was keyed without the model size, so the Large
run found Base's probability arrays, reused them, and skipped inference. The
tell was that all ten measurements matched to four decimal places, which two
architectures of different width cannot do.

Fixed in `aicd/models/droiddetect_baseline.py` (cache key now includes the
size) and guarded by `aicd/tests/test_droiddetect_cache.py`.
