"""Calibration: reliability diagrams, ECE, Brier, and the confidence-shape check.

Oedingen et al. found XGB+TF-IDF was almost perfectly calibrated while the DNN
and GMM models emitted almost nothing between 0.1 and 0.9 -- confidently wrong,
which is the worst failure mode for a system that gets used to accuse people.
The `check_confidence_shape` test below is that finding turned into an alarm.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from aicd import config as C
from aicd.data.splits import SLICES


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """ECE over the binary human-vs-rest decision the product acts on."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if not m.any():
            continue
        ece += m.sum() / n * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


def reliability_points(y_true: np.ndarray, p: np.ndarray, bins: int = 15):
    edges = np.linspace(0.0, 1.0, bins + 1)
    xs, ys, ns = [], [], []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum() < 5:
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y_true[m].mean()))
        ns.append(int(m.sum()))
    return xs, ys, ns


def check_confidence_shape(p: np.ndarray) -> dict:
    """Fraction of predictions in the honest middle band.

    A model emitting almost nothing between 0.1 and 0.9 is not calibrated,
    it is just loud.
    """
    mid = float(((p > 0.1) & (p < 0.9)).mean())
    return {
        "middle_band_fraction": mid,
        "verdict": "ok" if mid >= 0.05 else "OVERCONFIDENT",
    }


def fit_isotonic_per_language(df: pd.DataFrame, proba: np.ndarray, mask) -> dict:
    """One isotonic regressor per language, on the human-vs-rest probability."""
    from sklearn.isotonic import IsotonicRegression

    out = {}
    sub = df.loc[mask]
    p_machine = 1.0 - proba[:, 0]
    y = (sub["label"].to_numpy() != 0).astype(int)
    for lang in sub["language"].unique():
        lm = (sub["language"] == lang).to_numpy()
        if lm.sum() < 50 or len(np.unique(y[lm])) < 2:
            continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(p_machine[lm], y[lm])
        out[lang] = ir
    return out


def apply_isotonic(cals: dict, languages, p_machine: np.ndarray) -> np.ndarray:
    out = p_machine.copy()
    languages = np.asarray(languages)
    for lang, ir in cals.items():
        m = languages == lang
        if m.any():
            out[m] = ir.predict(p_machine[m])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--branch", default="b", choices=["a", "b", "stack"])
    args = ap.parse_args()
    cfg = C.load(args.config)
    art, rep = C.artifacts(cfg), C.reports(cfg)

    df = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet")
    prefix = {"a": "proba_a", "b": "proba_b", "stack": "proba_stack"}[args.branch]

    # Fit calibrators on val, where available.
    cals = {}
    vm = df["slice"] == "val"
    vp = art / f"{prefix}_val.npy"
    if vp.exists() and vm.any():
        cals = fit_isotonic_per_language(df, np.load(vp), vm)
        print(f"[calibration] fitted isotonic for {len(cals)} languages on val")

    report: dict = {"branch": args.branch, "slices": {}}
    print(f"\n{'slice':22s} {'n':>7s} {'ECE raw':>9s} {'ECE cal':>9s} "
          f"{'Brier':>8s} {'mid-band':>9s} {'verdict':>14s}")
    print("-" * 82)
    for s in SLICES:
        p = art / f"{prefix}_{s}.npy"
        m = df["slice"] == s
        if not p.exists() or not m.any():
            continue
        proba = np.load(p)
        sub = df.loc[m]
        if len(proba) != len(sub):
            continue

        y = (sub["label"].to_numpy() != 0).astype(int)
        pm = 1.0 - proba[:, 0]
        if len(np.unique(y)) < 2:
            continue

        ece_raw = expected_calibration_error(y, pm)
        pm_cal = apply_isotonic(cals, sub["language"], pm) if cals else pm
        ece_cal = expected_calibration_error(y, pm_cal)
        brier = float(np.mean((pm_cal - y) ** 2))
        shape = check_confidence_shape(pm)
        xs, ys, ns = reliability_points(y, pm_cal)

        report["slices"][s] = {
            "n": int(m.sum()), "ece_raw": ece_raw, "ece_calibrated": ece_cal,
            "brier": brier, **shape,
            "reliability": {"pred": xs, "obs": ys, "n": ns},
        }
        print(f"{s:22s} {int(m.sum()):>7,} {ece_raw:>9.4f} {ece_cal:>9.4f} "
              f"{brier:>8.4f} {shape['middle_band_fraction']:>9.3f} {shape['verdict']:>14s}")

    with open(rep / f"calibration_{args.branch}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if cals:
        import pickle
        with open(art / f"isotonic_{args.branch}.pkl", "wb") as f:
            pickle.dump(cals, f)

    bad = [s for s, v in report["slices"].items() if v["verdict"] != "ok"]
    if bad:
        print(f"\n[warn] overconfident on: {', '.join(bad)}")
        print("       Predictions cluster at 0 and 1. Do not surface raw")
        print("       probabilities from this branch to users.")
    print(f"\n[done] {rep / f'calibration_{args.branch}.json'}")


if __name__ == "__main__":
    main()
