"""The twin effect across three seeds, so it can be separated from training noise.

The twin previously rested on one run per arm, which cannot distinguish an
exposure effect from initialisation luck. Both arms were replicated at seeds 2
and 3 with `--train-seed`, leaving `cfg.project.seed` alone so the corpus and the
arm partition are identical across seeds; every run printed the same arm
checksum and refused to train otherwise.

The quantity is the paired difference per seed, f(D2_k) - f(D1_small_k), and what
matters is its mean and spread. Reporting the spread of each arm separately would
understate the precision, because the two arms move together across seeds.

    python -m aicd.eval.seed_twin
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "kaggle_runs" / "results" / "seeds"
OUT = ROOT / "aicd" / "eval" / "reports" / "seed_twin.json"

CONDS = ("s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
         "s4_unseen_domain", "s5_compound")
RUNS = {
    2: ("e1-seed-2-exposed-arm-d2", "e1-d1small-s2"),
    3: ("e1-seed-3-exposed-arm-d2", "e1-d1small-s3"),
}


def load(slug: str, pattern: str) -> dict:
    reps = list((SEEDS / slug / "extracted" / "reports").glob(pattern))
    if not reps:
        raise SystemExit(f"no report matching {pattern} under {slug}")
    r = json.loads(reps[0].read_text(encoding="utf-8"))["slices"]
    return {c: r[c] for c in CONDS if c in r}


def main() -> None:
    tc = json.loads((ROOT / "aicd" / "eval" / "reports"
                     / "twin_control.json").read_text(encoding="utf-8"))["conditions"]
    arms = {1: {c: {"d2": tc[c]["d2"], "d1small": tc[c]["d1small"],
                    "n": tc[c]["rows"]} for c in CONDS}}

    for seed, (d2_slug, d1s_slug) in RUNS.items():
        d2 = load(d2_slug, "branch_a_e1_d2*.json")
        d1s = load(d1s_slug, "branch_a_e1_d1small*.json")
        arms[seed] = {c: {"d2": d2[c]["macro_f1"], "d1small": d1s[c]["macro_f1"],
                          "n": d2[c]["n"]} for c in CONDS}

    # Every seed must have been scored on the same rows, or the differences are
    # not comparable. This is the check the arm checksum exists to make good.
    for c in CONDS:
        ns = {arms[s][c]["n"] for s in arms}
        if len(ns) != 1:
            raise SystemExit(f"{c}: seeds disagree on row count {ns}; the arms "
                             "are not the same partition and cannot be pooled")

    report = {
        "note": ("Paired difference D2 minus D1_small per seed. cfg.project.seed "
                 "is identical across runs, so the corpus and arm partition are "
                 "too; only weight initialisation differs."),
        "seeds": sorted(arms),
        "conditions": {},
    }
    for c in CONDS:
        diffs = [round(arms[s][c]["d2"] - arms[s][c]["d1small"], 4)
                 for s in sorted(arms)]
        mean, sd = st.mean(diffs), st.stdev(diffs)
        report["conditions"][c] = {
            "rows": arms[1][c]["n"],
            "d2": [arms[s][c]["d2"] for s in sorted(arms)],
            "d1small": [arms[s][c]["d1small"] for s in sorted(arms)],
            "differences": diffs,
            "mean": round(mean, 4),
            "sd": round(sd, 4),
            "range": round(max(diffs) - min(diffs), 4),
            # A mean many standard deviations from zero cannot be seed noise.
            "sd_from_zero": round(abs(mean) / sd, 1) if sd else None,
        }

    s5 = report["conditions"]["s5_compound"]
    s1 = report["conditions"]["s1_in_distribution"]
    report["headline"] = {
        "compound_mean": s5["mean"], "compound_sd": s5["sd"],
        "in_distribution_mean": s1["mean"], "in_distribution_sd": s1["sd"],
        "survives_replication": bool(abs(s5["mean"]) > 5 * s5["sd"]),
    }

    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"  {'condition':<24}{'seed1':>9}{'seed2':>9}{'seed3':>9}"
          f"{'mean':>9}{'sd':>8}{'sd from 0':>11}")
    for c, v in report["conditions"].items():
        d = v["differences"]
        sf = f"{v['sd_from_zero']:.1f}" if v["sd_from_zero"] else "-"
        print(f"  {c:<24}" + "".join(f"{x:>+9.4f}" for x in d)
              + f"{v['mean']:>+9.4f}{v['sd']:>8.4f}{sf:>11}")
    print(f"\n  -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
