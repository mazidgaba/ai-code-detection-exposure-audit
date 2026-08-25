"""Compare design variation against seed variation on the same footing.

The paper previously set the range of four design arms against the range of six
seeds. Range grows with sample size, so comparing a range over four to a range
over six is biased toward whichever has more draws, and the conclusion happened
to survive the bias rather than being established despite it. Standard deviation
does not have that problem, and the ratio of the two variances says directly how
much of the spread in the measured collapse is attributable to each factor.

    python -m aicd.eval.variance_components
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "aicd" / "eval" / "reports"
OUT = REP / "variance_components.json"


def main() -> None:
    ab = json.loads((REP / "e5_ablation.json").read_text(encoding="utf-8"))
    ss = json.loads((REP / "seed_sweep_six.json").read_text(encoding="utf-8"))

    design = []
    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("drop"), (int, float)):
                design.append(float(x["drop"]))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(ab)

    # The sweep stores per-seed condition vectors; the drop is S1 minus S5.
    seed = [round(v[0] - v[-1], 4) for v in ss["seeds"].values()]

    if len(design) < 2 or len(seed) < 2:
        raise SystemExit(f"need both factors: design={len(design)} seed={len(seed)}")

    d_sd, s_sd = st.stdev(design), st.stdev(seed)
    report = {
        "note": ("Standard deviation rather than range, because range grows with "
                 "the number of draws and the two factors have different counts."),
        "design": {"n": len(design), "values": [round(x, 4) for x in design],
                   "mean": round(st.mean(design), 4), "sd": round(d_sd, 4),
                   "range": round(max(design) - min(design), 4)},
        "seed": {"n": len(seed), "values": seed,
                 "mean": round(st.mean(seed), 4), "sd": round(s_sd, 4),
                 "range": round(max(seed) - min(seed), 4)},
        "sd_ratio_seed_over_design": round(s_sd / d_sd, 2),
        "variance_share_seed": round(s_sd ** 2 / (s_sd ** 2 + d_sd ** 2), 3),
        "seed_dominates": bool(s_sd > d_sd),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for k in ("design", "seed"):
        v = report[k]
        print(f"  {k:<8} n={v['n']}  mean={v['mean']:.4f}  sd={v['sd']:.4f}  "
              f"range={v['range']:.4f}")
    print(f"\n  seed sd / design sd     : {report['sd_ratio_seed_over_design']}")
    print(f"  share of variance = seed: {report['variance_share_seed']}")
    print(f"  seed dominates design   : {report['seed_dominates']}")
    print(f"  -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
