"""Paired comparisons and multiplicity, for the claims the paper makes twice.

Two reviewer findings.

**Seed variance as a paired quantity with an effect size.** The paper reported
the six-seed drop as a mean and a standard deviation. A standard deviation is
not an effect size, and with six runs a ratio of the drop to its own standard
deviation is itself unstable. What the claim needs is the paired drop, its
interval, and a standardised effect size stated with its degrees of freedom.

**Multiplicity on the generator comparison.** The independent corpus contains
three generators, and the paper singles out Gemini as detected worse than the
other two. That is two hypothesis tests chosen after seeing which generator
looked worst, and reporting two uncorrected p-values understates the chance of
finding one. Holm's step-down procedure controls the family-wise error rate
without assuming independence, which matters because both comparisons share the
Gemini sample.

Interval estimates are reported alongside, because a p-value says a difference
exists and an interval says how large it might be.

    python -m aicd.eval.comparisons
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "aicd" / "eval" / "reports"
CONDITIONS = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
              "s4_unseen_domain", "s5_compound"]


def two_proportion_z(k1, n1, k2, n2):
    """Pooled two-proportion z test, returning z and the two-sided p."""
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    return z, 2 * stats.norm.sf(abs(z))


def newcombe_difference(k1, n1, k2, n2, alpha=0.05):
    """Newcombe's hybrid-score interval for a difference of proportions.

    The Wald interval misbehaves near 0 and 1 and at unequal sample sizes;
    Newcombe's method combines two Wilson intervals and holds up where Wald
    does not. Reported because the size of the gap matters as much as its
    existence.
    """
    z = stats.norm.ppf(1 - alpha / 2)

    def wilson(k, n):
        p = k / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return (c - h) / d, (c + h) / d

    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return p1 - p2, lo, hi


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values, keyed as the input."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)      # enforce monotonicity
        out[k] = running
    return out


def seed_effect_size() -> dict:
    """The six-seed drop as a paired quantity with an effect size."""
    s = json.loads((REPORTS / "seed_sweep_six.json").read_text(encoding="utf-8"))
    drops = np.array([v[0] - v[-1] for v in s["seeds"].values()], dtype=float)
    n = len(drops)
    mean, sd = float(drops.mean()), float(drops.std(ddof=1))
    se = sd / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, n - 1)
    ci = (mean - t_crit * se, mean + t_crit * se)
    t_stat = mean / se
    # Cohen's d_z for a one-sample/paired design: the mean divided by the sd of
    # the differences, not by the standard error. Reported with df because at
    # n = 6 the estimate of sd is itself uncertain.
    d_z = mean / sd
    return {"n_seeds": n, "drops": drops.tolist(), "mean": mean, "sd": sd,
            "se": se, "ci95": list(ci), "t": t_stat, "df": n - 1,
            "p": float(2 * stats.t.sf(abs(t_stat), n - 1)), "cohens_dz": d_z}


def main() -> None:
    out = {}

    # ---- 4.3 seed variance -------------------------------------------------
    sv = seed_effect_size()
    out["seed_drop"] = sv
    print("Six-seed S1-to-S5 drop, treated as a paired quantity")
    print(f"  drops        {'  '.join(f'{d:.4f}' for d in sorted(sv['drops']))}")
    print(f"  mean         {sv['mean']:.4f}   sd {sv['sd']:.4f}   se {sv['se']:.4f}")
    print(f"  95% CI       [{sv['ci95'][0]:.4f}, {sv['ci95'][1]:.4f}]  (t, df={sv['df']})")
    print(f"  t({sv['df']})        {sv['t']:.1f}   p = {sv['p']:.2e}")
    print(f"  Cohen's d_z  {sv['cohens_dz']:.1f}")
    print("  The interval excludes zero by far more than its own width. We report")
    print("  d_z rather than a drop-to-sd ratio dressed as one, and state df,")
    print("  because at six runs the sd is itself imprecisely estimated.")

    # ---- 4.4 generator comparisons with multiplicity ----------------------
    ie = json.loads((REPORTS / "independent_eval.json").read_text(encoding="utf-8"))
    pg = ie["per_generator"]
    counts = {g: (int(round(v["recall_as_nonhuman"] * v["n"])), int(v["n"]))
              for g, v in pg.items()}

    target = "gemini"
    others = [g for g in counts if g != target]
    raw, detail = {}, {}
    for g in others:
        k1, n1 = counts[target]
        k2, n2 = counts[g]
        z, p = two_proportion_z(k1, n1, k2, n2)
        diff, lo, hi = newcombe_difference(k1, n1, k2, n2)
        raw[g] = p
        detail[g] = {"z": z, "p_raw": p, "difference": diff, "ci95": [lo, hi],
                     "n_target": n1, "n_other": n2}
    adj = holm(raw)

    print(f"\nGemini against the other two generators, recall as non-human")
    print(f"  {'comparison':22s} {'diff':>8s} {'95% CI':>20s} {'z':>7s} "
          f"{'p':>10s} {'Holm p':>10s}")
    print("  " + "-" * 82)
    for g in sorted(others):
        d = detail[g]
        d["p_holm"] = adj[g]
        lo, hi = d["ci95"]
        print(f"  {'gemini - ' + g:22s} {d['difference']:+8.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] {d['z']:7.2f} {d['p_raw']:10.2e} "
              f"{d['p_holm']:10.2e}")
    out["generator_comparison"] = {"target": target, "counts": counts,
                                   "tests": detail, "correction": "holm",
                                   "family_size": len(others)}
    survives = all(d["p_holm"] < 0.05 for d in detail.values())
    out["generator_comparison"]["all_survive_holm"] = survives
    print()
    if survives:
        print("  Both survive Holm correction. The generator was chosen after seeing")
        print("  which looked worst, so correcting was necessary; it changes the")
        print("  p-values and not the conclusion.")
    else:
        print("  At least one comparison does not survive correction. The paper must")
        print("  not describe that generator as detected significantly worse.")

    dest = REPORTS / "comparisons.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
