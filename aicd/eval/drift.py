"""Drift monitoring.

A detector goes stale in months: new model releases change the input
distribution underneath it, and nothing in the request tells you that has
happened. Population Stability Index over the score distribution is the cheap
early warning.

PSI reading (industry convention):
    < 0.10  stable
    0.10-0.25  moderate shift, investigate
    > 0.25  significant shift, retrain
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from aicd import config as C
from aicd.data.splits import SLICES

BINS = 10


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = BINS) -> float:
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / max(len(expected), 1)
    a = np.histogram(actual, edges)[0] / max(len(actual), 1)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def verdict(v: float) -> str:
    return "stable" if v < 0.10 else ("moderate" if v < 0.25 else "SIGNIFICANT")


def baseline_path(cfg) -> Path:
    return C.artifacts(cfg) / "drift_baseline.json"


def store_path(cfg) -> Path:
    return C.artifacts(cfg) / "drift_history.jsonl"


def build_baseline(cfg, branch: str = "b") -> dict:
    """Reference distribution = the slice the thresholds were fitted on."""
    art = C.artifacts(cfg)
    df = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet")
    prefix = f"proba_{branch}"
    src = "val" if (art / f"{prefix}_val.npy").exists() else "s1_in_distribution"
    proba = np.load(art / f"{prefix}_{src}.npy")
    sub = df[df["slice"] == src]
    pm = 1.0 - proba[:, 0]

    base = {"source": src, "overall": pm.tolist()[:20000], "by_language": {}}
    for lang, g in sub.groupby("language"):
        idx = sub.index.get_indexer(g.index)
        if len(idx) >= 100:
            base["by_language"][lang] = pm[idx].tolist()[:5000]
    with open(baseline_path(cfg), "w", encoding="utf-8") as f:
        json.dump(base, f)
    print(f"[drift] baseline from '{src}': overall n={len(base['overall']):,}, "
          f"languages={sorted(base['by_language'])}")
    return base


def check(cfg, scores: np.ndarray, languages: list[str] | None = None) -> dict:
    p = baseline_path(cfg)
    if not p.exists():
        raise SystemExit("no baseline; run with --build-baseline first")
    with open(p, "r", encoding="utf-8") as f:
        base = json.load(f)

    out = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "n": int(len(scores)),
        "overall_psi": psi(np.array(base["overall"]), scores),
        "by_language": {},
    }
    out["overall_verdict"] = verdict(out["overall_psi"])

    if languages is not None:
        languages = np.asarray(languages)
        for lang, ref in base["by_language"].items():
            m = languages == lang
            if m.sum() >= 50:
                v = psi(np.array(ref), scores[m])
                out["by_language"][lang] = {"psi": v, "n": int(m.sum()), "verdict": verdict(v)}

    with open(store_path(cfg), "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--branch", default="b")
    ap.add_argument("--build-baseline", action="store_true")
    ap.add_argument("--check-slice", default=None,
                    help="simulate a drift check using one evaluation slice")
    args = ap.parse_args()
    cfg = C.load(args.config)

    if args.build_baseline:
        build_baseline(cfg, args.branch)
        return

    if args.check_slice:
        art = C.artifacts(cfg)
        df = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet")
        slices = (list(SLICES) if args.check_slice == "all"
                  else [args.check_slice])

        dest = C.reports(cfg) / "drift.json"
        report = (json.loads(dest.read_text(encoding="utf-8"))
                  if dest.exists() else {"branch": args.branch, "slices": {}})

        for name in slices:
            p = art / f"proba_{args.branch}_{name}.npy"
            if not p.exists():
                print(f"[drift] {name}: no probabilities on disk, skipped")
                continue
            proba = np.load(p)
            sub = df[df["slice"] == name]
            r = check(cfg, 1.0 - proba[:, 0], sub["language"].tolist())

            print(f"[drift] slice={name} n={r['n']:,}")
            print(f"  overall PSI = {r['overall_psi']:.4f}  -> {r['overall_verdict']}")
            for lang, v in sorted(r["by_language"].items(), key=lambda kv: -kv[1]["psi"]):
                print(f"  {lang:12s} PSI={v['psi']:.4f} n={v['n']:>6,}  {v['verdict']}")

            # Persist it. The PSI numbers are quoted in the paper, so leaving
            # them only on stdout means nothing can audit them later.
            report["slices"][name] = r

        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n-> {dest}")
        print("A SIGNIFICANT reading on an OOD slice is the expected result --")
        print("that is the alarm doing its job, not a false positive.")
        return

    ap.error("pass --build-baseline or --check-slice")


if __name__ == "__main__":
    main()
