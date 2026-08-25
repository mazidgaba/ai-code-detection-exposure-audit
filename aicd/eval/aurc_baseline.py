"""AURC against the baseline that says what a given AURC is worth.

The manuscript reports the area under the risk-coverage curve for each
condition. On its own that number is uninterpretable, because AURC depends on
the error rate as much as on the ordering: a detector with a 60% error rate has
a large AURC even if its confidence ordering is perfect, and a detector with a
2% error rate has a small one even if its ordering is worthless. Comparing
0.0097 on S1 against 0.8234 on S5 therefore conflates two different things.

Two reference points fix that.

**Random ranker.** Shuffle the confidence ordering. Selective risk is then the
overall error rate at every coverage, so AURC equals that error rate. This is
what a detector achieves by knowing nothing about which of its own predictions
are reliable, and it is the number an AURC must beat to mean anything.

**Oracle ranker.** Order every correct prediction ahead of every incorrect one.
This is the lowest AURC attainable at that error rate, and it is not zero.

Between them the interval [oracle, random] is where any real detector must fall,
and the position within that interval is the part attributable to the ordering
rather than to the error rate. We report it as

    normalised AURC = (AURC - oracle) / (random - oracle)

which is 0 for a perfect ordering and 1 for a worthless one, whatever the error
rate underneath.

    python -m aicd.eval.aurc_baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "aicd" / "eval" / "reports"
RUNS = ROOT / "kaggle_runs" / "results"
KEYS = ROOT / "aicd" / "artifacts" / "keys" / "eval_row_keys.parquet"
CONDITIONS = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
              "s4_unseen_domain", "s5_compound"]
HUMAN = 0


def risk_coverage(wrong: np.ndarray, conf: np.ndarray):
    """Risk-coverage curve for a confidence ordering, most confident first."""
    order = np.argsort(-conf, kind="stable")
    w = wrong[order]
    k = np.arange(1, len(w) + 1)
    return k / len(w), np.cumsum(w) / k


def aurc(wrong: np.ndarray, conf: np.ndarray) -> float:
    cov, risk = risk_coverage(wrong, conf)
    return float(np.trapezoid(risk, cov))


def aurc_random(wrong: np.ndarray, rng: np.random.Generator, reps: int = 32) -> float:
    """Expected AURC under a shuffled ordering.

    Averaged over repeats rather than asserted to equal the error rate: the
    finite-sample curve wobbles at low coverage, and the average is what a real
    random ranker would deliver.
    """
    return float(np.mean([aurc(wrong, rng.random(len(wrong))) for _ in range(reps)]))


def random_null(wrong: np.ndarray, observed: float, rng: np.random.Generator,
                reps: int = 500) -> dict:
    """Where the observed AURC falls in the distribution of shuffled orderings.

    A normalised AURC above 1 looks like an ordering worse than chance, and on a
    small condition that reading does not survive contact with the null: the
    shuffled distribution is wide enough that a value a few percent above its
    mean is unremarkable. This returns the null so the claim can be stated at
    the strength the evidence supports rather than at the strength the point
    estimate suggests.
    """
    null = np.array([aurc(wrong, rng.random(len(wrong))) for _ in range(reps)])
    sd = float(null.std(ddof=1))
    return {"null_mean": float(null.mean()), "null_sd": sd,
            "z": float((observed - null.mean()) / sd) if sd > 0 else float("nan"),
            "p_worse_than_random": float((null >= observed).mean()),
            "reps": reps}


def aurc_oracle(wrong: np.ndarray) -> float:
    """Lowest AURC attainable at this error rate: all correct rows first."""
    return aurc(wrong, (~wrong.astype(bool)).astype(float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default=str(RUNS / "e5" / "arrays"))
    ap.add_argument("--prefix", default="proba_a_e5_large")
    ap.add_argument("--report", default=str(RUNS / "e5" / "branch_a_e5_large.json"))
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    if not KEYS.exists():
        raise SystemExit(f"{KEYS} not found; produced by aicd.eval.contamination")
    key = pd.read_parquet(KEYS)
    rep = json.loads(Path(args.report).read_text(encoding="utf-8"))["slices"]
    rng = np.random.default_rng(args.seed)

    out = {"model": Path(args.report).stem, "conditions": {}}
    print(f"{'condition':22s} {'err':>7s} {'AURC':>8s} {'oracle':>8s} "
          f"{'random':>8s} {'normalised':>11s} {'r(c,ok)':>7s}")
    print("-" * 80)
    for c in CONDITIONS:
        f = Path(args.arrays) / f"{args.prefix}_{c}.npy"
        if not f.exists():
            continue
        y = key[key["slice"] == c]["label"].to_numpy()
        p = np.load(f)
        if len(y) != len(p):
            raise SystemExit(f"{c}: {len(p)} scored rows against {len(y)} labels")
        pred = p.argmax(1)

        # Verify alignment before trusting anything computed from it.
        from aicd.eval.resample import macro_f1
        if abs(macro_f1(y, pred) - rep[c]["macro_f1"]) > 1e-6:
            raise SystemExit(f"{c}: labels do not reproduce the reported macro-F1")

        wrong = (pred != y).astype(float)
        conf = p.max(1)
        a = aurc(wrong, conf)
        o = aurc_oracle(wrong)
        r = aurc_random(wrong, rng)
        norm = (a - o) / (r - o) if r > o else float("nan")
        from scipy.stats import pointbiserialr
        rb, pv = pointbiserialr((~wrong.astype(bool)).astype(int), conf)
        nul = random_null(wrong, a, rng)
        out["conditions"][c] = {"n": int(len(y)), "error_rate": float(wrong.mean()),
                                "aurc": a, "aurc_oracle": o, "aurc_random": r,
                                "aurc_normalised": norm,
                                "conf_correct_r": float(rb), "conf_correct_p": float(pv),
                                "null": nul,
                                "beats_random": bool(nul["p_worse_than_random"] > 0.05
                                                     and norm < 1.0)}
        print(f"{c:22s} {wrong.mean():7.4f} {a:8.4f} {o:8.4f} {r:8.4f} {norm:11.4f} "
              f"{rb:+7.3f}")

    print()
    vals = [(c, v["aurc_normalised"]) for c, v in out["conditions"].items()]
    if vals:
        raw = [out["conditions"][c]["aurc"] for c in out["conditions"]]
        nrm = [v for _, v in vals]
        print(f"Raw AURC spans {min(raw):.4f} to {max(raw):.4f}, a factor of "
              f"{max(raw)/max(min(raw), 1e-9):.0f}.")
        print(f"Normalised it spans {min(nrm):.4f} to {max(nrm):.4f}.")
        print()
        print("The raw spread is mostly the error rate changing, not the ordering.")
        print("The normalised figure is the part that belongs to the detector's")
        print("own sense of when it is unreliable, and it is what an abstention")
        print("mechanism can act on.")
        worst = max(vals, key=lambda kv: kv[1])
        print(f"\nOrdering is least useful on {worst[0]} at {worst[1]:.4f} of the way")
        print("from a perfect ordering to a worthless one.")

    dest = REPORTS / "aurc_baseline.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
