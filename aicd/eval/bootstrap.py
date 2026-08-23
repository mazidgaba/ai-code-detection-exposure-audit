"""Bootstrap confidence intervals for every headline metric.

The paper currently reports point estimates from a single run. That is a
standard reviewer objection, and it bites hardest exactly where the paper's
claims are strongest: S5 has 1,526 rows, so its 0.209 macro-F1 carries real
sampling uncertainty, and the operating-point rates on the human subset rest on
fewer rows still.

This resamples the evaluation set with replacement and reports percentile
intervals. It quantifies sampling uncertainty on a fixed model; it does not
capture training variance, which would need multiple seeds. The distinction is
stated in the output so the paper cannot overclaim it.

Usage:
    python -m aicd.eval.bootstrap --config cpu.yaml --branch b --n 2000
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from aicd import config as C
from aicd.data.splits import SLICES

HUMAN = 0


def metrics_once(y: np.ndarray, proba: np.ndarray, t_hi: float, t_lo: float) -> dict:
    pred = proba.argmax(1)
    p_machine = 1.0 - proba[:, HUMAN]
    out = {
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
    }
    yb = (y != HUMAN).astype(int)
    out["binary_auc"] = (roc_auc_score(yb, p_machine)
                         if len(np.unique(yb)) > 1 else np.nan)

    human = y == HUMAN
    out["human_fpr_argmax"] = float((pred[human] != HUMAN).mean()) if human.any() else np.nan

    called = (p_machine >= t_hi) | (p_machine <= t_lo)
    hc = human & called
    out["human_fpr_policy"] = (float((p_machine[hc] >= t_hi).mean())
                               if hc.any() else np.nan)
    out["coverage"] = float(called.mean())
    return out


def bootstrap_slice(y, proba, t_hi, t_lo, n_boot, rng) -> dict:
    point = metrics_once(y, proba, t_hi, t_lo)
    keys = list(point)
    draws = {k: [] for k in keys}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], proba[idx]
        if len(np.unique(yb)) < 2:
            continue
        m = metrics_once(yb, pb, t_hi, t_lo)
        for k in keys:
            if not np.isnan(m[k]):
                draws[k].append(m[k])

    res = {}
    for k in keys:
        d = np.asarray(draws[k], dtype=float)
        if d.size < 20:
            res[k] = {"point": point[k], "lo": np.nan, "hi": np.nan, "n_draws": int(d.size)}
            continue
        res[k] = {
            "point": float(point[k]),
            "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)),
            "se": float(d.std(ddof=1)),
            "n_draws": int(d.size),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu.yaml")
    ap.add_argument("--branch", default="b")
    ap.add_argument("--n", type=int, default=2000, help="bootstrap resamples")
    args = ap.parse_args()

    cfg = C.load(args.config)
    art, rep = C.artifacts(cfg), C.reports(cfg)
    df = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet")

    pol = json.loads((rep / "policy_b.json").read_text(encoding="utf-8"))
    t_hi, t_lo = pol["thresholds"]["t_high"], pol["thresholds"]["t_low"]

    rng = np.random.default_rng(cfg.project.seed)
    out = {"n_boot": args.n, "thresholds": {"t_high": t_hi, "t_low": t_lo},
           "note": ("percentile bootstrap over the evaluation set; captures "
                    "sampling uncertainty for a fixed model, not training variance"),
           "slices": {}}

    hdr = f"{'slice':22s} {'n':>7s}  {'macro-F1 [95% CI]':>26s}  {'human FPR @policy':>24s}"
    print(hdr)
    print("-" * len(hdr))

    for s in SLICES:
        f = art / f"proba_{args.branch}_{s}.npy"
        if not f.exists():
            continue
        proba = np.load(f)
        y = df.loc[df["slice"] == s, "label"].to_numpy()
        if len(y) != len(proba):
            continue
        r = bootstrap_slice(y, proba, t_hi, t_lo, args.n, rng)
        out["slices"][s] = r
        mf, hf = r["macro_f1"], r["human_fpr_policy"]
        print(f"{s:22s} {len(y):>7,}  "
              f"{mf['point']:.4f} [{mf['lo']:.4f}, {mf['hi']:.4f}]  "
              f"{hf['point']:.4f} [{hf['lo']:.4f}, {hf['hi']:.4f}]")

    dest = rep / "bootstrap.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> {dest}")

    # The comparison the paper actually rests on.
    s1 = out["slices"].get("s1_in_distribution", {}).get("human_fpr_policy")
    s4 = out["slices"].get("s4_unseen_domain", {}).get("human_fpr_policy")
    if s1 and s4 and not np.isnan(s1["lo"]) and not np.isnan(s4["lo"]):
        sep = s4["lo"] > s1["hi"]
        print("\nHeadline claim, human FPR at the fitted operating point:")
        print(f"  S1  {s1['point']:.4f}  95% CI [{s1['lo']:.4f}, {s1['hi']:.4f}]")
        print(f"  S4  {s4['point']:.4f}  95% CI [{s4['lo']:.4f}, {s4['hi']:.4f}]")
        print(f"  intervals {'do not overlap' if sep else 'OVERLAP'} -> "
              f"{'separation is not a sampling artefact' if sep else 'claim is weaker than stated'}")


if __name__ == "__main__":
    main()
