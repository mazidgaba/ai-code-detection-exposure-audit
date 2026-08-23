"""Run the pipeline end to end, or from a chosen stage.

    python -m aicd.scripts.run_pipeline --from features
    python -m aicd.scripts.run_pipeline --only branch_b calibration policy
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

STAGES = [
    ("download",    ["-m", "aicd.data.download", "--train-shards", "3"]),
    ("normalize",   ["-m", "aicd.data.normalize"]),
    ("filter",      ["-m", "aicd.data.filter"]),
    ("splits",      ["-m", "aicd.data.splits"]),
    ("tests",       ["-m", "pytest", "aicd/tests/", "-q"]),
    ("stats",       ["-m", "aicd.data.stats"]),
    ("features",    ["-m", "aicd.features.build"]),
    ("branch_b",    ["-m", "aicd.models.xgb"]),
    ("shap",        ["-m", "aicd.models.shap_report"]),
    ("calibration", ["-m", "aicd.eval.calibration", "--branch", "b"]),
    ("policy",      ["-m", "aicd.serve.policy", "--branch", "b"]),
    ("drift_base",  ["-m", "aicd.eval.drift", "--build-baseline"]),
]

# Stages that need a GPU to be worth running; skipped unless --with-neural.
NEURAL = [
    ("branch_a",    ["-m", "aicd.models.modernbert_triplet", "--resample"]),
    ("branch_c",    ["-m", "aicd.models.fastdetect"]),
    ("stacker",     ["-m", "aicd.models.stacker"]),
]


def run(name: str, args: list[str], config: str) -> bool:
    cmd = [sys.executable, "-u"] + args
    if "pytest" not in args:
        cmd += ["--config", config]
    print(f"\n{'=' * 70}\n[stage] {name}\n{'=' * 70}", flush=True)
    t = time.time()
    r = subprocess.run(cmd)
    dt = time.time() - t
    ok = r.returncode == 0
    print(f"[stage] {name}: {'ok' if ok else 'FAILED'} in {dt / 60:.1f} min")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--with-neural", action="store_true",
                    help="also run branches A and C (needs a GPU to be practical)")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    stages = STAGES + (NEURAL if args.with_neural else [])
    if args.only:
        stages = [s for s in stages if s[0] in set(args.only)]
    elif args.start:
        names = [s[0] for s in stages]
        if args.start not in names:
            ap.error(f"unknown stage {args.start}; choices: {names}")
        stages = stages[names.index(args.start):]

    failed = []
    for name, cmd in stages:
        if not run(name, cmd, args.config):
            failed.append(name)
            if not args.keep_going:
                print(f"\n[pipeline] stopped at '{name}'. Use --keep-going to continue.")
                return 1

    print(f"\n[pipeline] {'FAILED: ' + ', '.join(failed) if failed else 'all stages ok'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
