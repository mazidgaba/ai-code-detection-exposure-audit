"""End-to-end smoke test on synthetic rows.

Exercises every stage with stub-sized data so wiring breaks surface in seconds
rather than after a 40-minute training run. Writes eval/reports/smoke.json.
"""
from __future__ import annotations

import json
import sys
import time
import traceback

import numpy as np
import pandas as pd

from aicd import config as C

HUMAN = [
    "def f(x):\n  if x>0: return x\n  return -x\n",
    "def s(a,b):\n  t=a+b\n  return t\n",
    "def q(l):\n  if len(l)<2: return l\n  p=l[0]\n  return q([x for x in l[1:] if x<p])+[p]\n",
    "def r(n):\n  a,b=0,1\n  while a<n:\n    a,b=b,a+b\n  return a\n",
]
MACHINE = [
    "def compute_absolute_value(number):\n\n    if number > 0:\n\n        return number\n\n    return -number\n",
    "def calculate_sum(first_value, second_value):\n\n    total = first_value + second_value\n\n    return total\n",
    "def quick_sort(items):\n\n    if len(items) < 2:\n\n        return items\n\n    pivot = items[0]\n\n    return quick_sort([i for i in items[1:] if i < pivot]) + [pivot]\n",
    "def fibonacci_below(limit):\n\n    current, following = 0, 1\n\n    while current < limit:\n\n        current, following = following, current + following\n\n    return current\n",
]


def synth(n: int = 400) -> pd.DataFrame:
    rows = []
    for i in range(n):
        lab = i % 4
        src = HUMAN if lab == 0 else MACHINE
        code = src[i % len(src)]
        if lab == 2:                       # hybrid: half human, half machine
            code = HUMAN[i % 4] + MACHINE[i % 4]
        if lab == 3:                       # adversarial: machine, de-styled
            code = "\n".join(l for l in MACHINE[i % 4].split("\n") if l.strip())
        code += f"\n# variant {i}\n" if lab else f"\n#{i}\n"
        rows.append({
            "code": code, "label": lab, "language": "python",
            "source": ["TACO", "LEETCODE", "STARCODER_DATA"][i % 3],
            "domain": "algorithmic",
            "model_family": ["human", "qwen", "microsoft", "meta-llama"][lab],
            "generator": "synthetic", "generation_mode": "SMOKE",
            "orig_split": "train", "problem_id": f"P{i // 4}",
            # Slice must be independent of label, or train sees only some
            # classes and XGBoost refuses to fit. (This exact bug is why the
            # smoke test exists.)
            "slice": ["train", "train", "val", "s1_in_distribution"][(i // 4) % 4],
        })
    df = pd.DataFrame(rows)
    df["split"] = np.where(df["slice"].isin(["train", "val"]), df["slice"], "test")
    return df


def main() -> int:
    t0 = time.time()
    cfg = C.load("smoke.yaml")
    result = {"stages": {}, "ok": True}

    def stage(name, fn):
        try:
            out = fn()
            result["stages"][name] = {"ok": True, **(out or {})}
            print(f"  [ok]   {name}")
        except Exception as e:
            result["stages"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            result["ok"] = False
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("[smoke] synthetic corpus")
    df = synth()
    print(f"  {len(df)} rows, labels={df['label'].value_counts().sort_index().tolist()}")

    # --- features
    def _features():
        from aicd.features.build import dense_features, select_columns
        X = dense_features(df)
        cols = select_columns(X, cfg.features.max_missing_fraction,
                              per_language=df["language"])
        return {"raw_width": int(X.shape[1]), "kept": len(cols)}
    stage("features", _features)

    # --- branch B
    def _xgb():
        from aicd.features.build import dense_features
        from aicd.models import xgb as XB
        dense = dense_features(df).astype(np.float32)
        vec, clf, dcols = XB.fit(cfg, df, dense, save=False)
        m = df["slice"] == "s1_in_distribution"
        p = XB.predict(vec, clf, dcols, df.loc[m, "code"], dense.loc[m])
        from sklearn.metrics import f1_score
        f1 = f1_score(df.loc[m, "label"], p.argmax(1), average="macro", zero_division=0)
        return {"n_test": int(m.sum()), "macro_f1": round(float(f1), 4)}
    stage("branch_b_xgb", _xgb)

    # --- metrics
    def _metrics():
        from aicd.eval.metrics import slice_metrics
        y = np.array([0, 1, 2, 3] * 10)
        p = np.eye(4)[y] * 0.7 + 0.1
        m = slice_metrics(y, p)
        assert set(m) >= {"macro_f1", "accuracy", "confusion"}
        return {"keys": len(m)}
    stage("metrics", _metrics)

    # --- calibration
    def _calib():
        from aicd.eval.calibration import check_confidence_shape, expected_calibration_error
        y = np.random.default_rng(0).integers(0, 2, 500)
        p = np.clip(y * 0.6 + np.random.default_rng(1).normal(0.2, 0.15, 500), 0, 1)
        return {"ece": round(expected_calibration_error(y, p), 4),
                "shape": check_confidence_shape(p)["verdict"]}
    stage("calibration", _calib)

    # --- policy
    def _policy():
        from aicd.serve.policy import choose_threshold, decide
        y = np.array([0] * 100 + [1] * 100)
        p = np.concatenate([np.linspace(0, 0.6, 100), np.linspace(0.4, 1, 100)])
        t = choose_threshold(y, p, 0.01)
        d = decide(np.array([0.1, 0.7, 0.1, 0.1]), {"t_high": t, "t_low": t - 0.15})
        assert d.confidence_note, "confidence_note must never be empty"
        return {"t_high": round(t, 4), "decision": d.decision}
    stage("policy", _policy)

    # --- attribution
    def _attr():
        from aicd.serve.attribution import (make_synthetic_hybrid, per_line_scores,
                                            score_windows, top_regions)
        code = HUMAN[2] * 6
        hy, truth = make_synthetic_hybrid(code)
        rng = np.random.default_rng(3)
        wins, pm = score_windows(code, lambda cs: rng.random((len(cs), 4)), window=8)
        ls = per_line_scores(code, wins, pm)
        return {"windows": len(wins), "regions": len(top_regions(ls, 0.5)),
                "synthetic_hybrid": hy is not None}
    stage("attribution", _attr)

    # --- drift
    def _drift():
        from aicd.eval.drift import psi, verdict
        rng = np.random.default_rng(5)
        a = rng.normal(0.3, 0.1, 2000)
        v_same = psi(a, rng.normal(0.3, 0.1, 2000))
        v_shift = psi(a, rng.normal(0.7, 0.1, 2000))
        assert v_shift > v_same, "PSI must react to a shifted distribution"
        return {"psi_same": round(v_same, 4), "psi_shifted": round(v_shift, 4),
                "verdict_shifted": verdict(v_shift)}
    stage("drift", _drift)

    result["elapsed_s"] = round(time.time() - t0, 2)
    dest = C.reports(cfg) / "smoke.json"
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[smoke] {'PASS' if result['ok'] else 'FAIL'} in {result['elapsed_s']}s -> {dest}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
