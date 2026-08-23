"""E5: does any design choice in Branch A account for the collapse?

Four arms vary one thing each against the reference configuration:

    triplet0     triplet loss removed entirely (weight 0.0)
    triplet02    triplet weight lowered from the reference to 0.2
    proj256      projection head widened to 256 dimensions
    large        ModernBERT-large in place of ModernBERT-base

The question is whether the S1 -> S5 collapse is a property of the task or an
artefact of how this particular detector was built. If some arm removed the
collapse, the paper's claim would be about our architecture rather than about
withholding, and the contribution would evaporate.

## Why this module exists rather than a table of deltas

Each arm's notebook printed its drop against a single number, 0.6599, labelled
"reference". That number is real -- it is seed 20260818 in
`seed_sweep_six.json` -- but it is the **largest of the six seed drops**
(the six are 0.6159 to 0.6599, mean 0.6391). Comparing every arm to the maximum
of a noisy reference makes every arm look like it shrinks the collapse, which
is a selection artefact and not a finding.

The sound comparison is against the seed *distribution*. An arm has done
something only if its drop falls outside the spread the reference configuration
already produces when nothing changes but the random seed. This module makes
that comparison and reports it either way.

    python -m aicd.eval.e5_ablation
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "aicd" / "eval" / "reports"
RUNS = ROOT / "kaggle_runs" / "results"

CONDITIONS = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
              "s4_unseen_domain", "s5_compound"]

# Two arms ran on Kaggle accounts that have since been rotated away, and only
# their kernel logs were retained locally. The logs carry the full per-slice
# table, so the macro-F1 figures are primary; the probability arrays are not
# recoverable without re-granting access to those accounts. Provenance is
# recorded per arm so the difference is visible rather than papered over.
ARMS = [
    ("triplet0", "triplet loss removed", RUNS / "e5" / "branch_a_e5_triplet0.json"),
    ("triplet02", "triplet weight 0.2", RUNS / "e5_triplet02_kernel.log"),
    ("proj256", "projection head 256", RUNS / "e5_proj256_kernel.log"),
    ("large", "ModernBERT-large", RUNS / "e5" / "branch_a_e5_large.json"),
]

ROW = re.compile(
    r"^(s[1-5]_[a-z_]+)\s+([\d,]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
    re.M)


def _text(path: Path) -> str:
    raw = io.open(path, encoding="utf-8", errors="ignore").read()
    try:
        rows = json.loads(raw)
        if isinstance(rows, list):
            return "\n".join(r.get("data", "") for r in rows if isinstance(r, dict))
    except Exception:
        pass
    return raw


DROP = re.compile(r"S1 -> S5:\s*[\d.]+ -> [\d.]+\s*\(drop ([\d.]+)\)")


def from_log(path: Path) -> tuple[dict, float]:
    """Recover the per-slice macro-F1 table from a Kaggle kernel log.

    The evaluation table is printed twice (once by the eval call, once by the
    notebook's own summary); both copies are identical, and a disagreement
    between them means the log is not what we think it is, so it raises.

    The drop is taken from the log's own summary line rather than recomputed
    from the table, because the table is rounded to four places and the summary
    is not. The two differ by up to 1e-4, which is immaterial to the finding but
    would otherwise put two different numbers for the same quantity into the
    record. They are cross-checked here at the precision the table supports.
    """
    t = _text(path)
    found: dict[str, float] = {}
    for m in ROW.finditer(t):
        cond, f1 = m.group(1), float(m.group(3))
        if cond in found and abs(found[cond] - f1) > 1e-9:
            raise SystemExit(f"{path.name}: conflicting macro-F1 for {cond}")
        found[cond] = f1
    missing = [c for c in CONDITIONS if c not in found]
    if missing:
        raise SystemExit(f"{path.name}: no macro-F1 for {missing}")
    d = DROP.findall(t)
    if not d:
        raise SystemExit(f"{path.name}: no drop summary line")
    drop = float(d[-1])
    table_drop = found["s1_in_distribution"] - found["s5_compound"]
    if abs(drop - table_drop) > 2e-4:
        raise SystemExit(f"{path.name}: drop line {drop} contradicts table "
                         f"{table_drop:.4f}")
    return {c: found[c] for c in CONDITIONS}, drop


def from_report(path: Path) -> tuple[dict, float]:
    r = json.loads(path.read_text(encoding="utf-8"))
    by = {c: float(r["slices"][c]["macro_f1"]) for c in CONDITIONS}
    return by, by["s1_in_distribution"] - by["s5_compound"]


def _twin_drops() -> tuple[float, float] | None:
    """The E1 twin control's two drops, for scale: held-out arm, exposed arm."""
    a = RUNS / "e1" / "branch_a_e1_d1small.json"
    b = RUNS / "e1-d2" / "results" / "reports" / "branch_a_e1_d2.json"
    if not (a.exists() and b.exists()):
        return None
    return from_report(a)[1], from_report(b)[1]


def main() -> None:
    seeds = json.loads((REPORTS / "seed_sweep_six.json").read_text(encoding="utf-8"))
    drops = {k: v[0] - v[-1] for k, v in seeds["seeds"].items()}
    lo, hi = min(drops.values()), max(drops.values())
    mean = seeds["drop_mean"]
    ci = seeds["ci95"]

    print("Reference configuration, six seeds")
    print(f"  drops      {'  '.join(f'{d:.4f}' for d in sorted(drops.values()))}")
    print(f"  range      {lo:.4f} to {hi:.4f}")
    print(f"  mean       {mean:.4f}   95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    print()
    print("The single figure each arm's notebook compared against, 0.6599, is the")
    print("largest of these six. Comparing to it flatters every arm. The range is")
    print("the honest yardstick.")
    print()

    out = {"reference": {"seed_drops": drops, "drop_mean": mean, "ci95": ci,
                         "drop_min": lo, "drop_max": hi},
           "arms": {}}

    print(f"{'arm':12s} {'what changed':22s} {'S1':>7s} {'S5':>7s} {'drop':>7s} "
          f"{'vs seed range':>16s}")
    print("-" * 76)
    for tag, what, path in ARMS:
        if not path.exists():
            raise SystemExit(f"missing source for arm {tag}: {path}")
        by, drop = from_report(path) if path.suffix == ".json" else from_log(path)
        s1, s5 = by["s1_in_distribution"], by["s5_compound"]
        if drop > hi:
            verdict = "above"
        elif drop < lo:
            verdict = "below"
        else:
            verdict = "inside"
        out["arms"][tag] = {
            "what_changed": what,
            "source": str(path.relative_to(ROOT)).replace("\\", "/"),
            "provenance": "report" if path.suffix == ".json" else "kernel log",
            "by_condition": by, "drop": drop, "vs_seed_range": verdict,
        }
        print(f"{tag:12s} {what:22s} {s1:7.4f} {s5:7.4f} {drop:7.4f} {verdict:>16s}")

    arm_drops = [a["drop"] for a in out["arms"].values()]
    a_lo, a_hi = min(arm_drops), max(arm_drops)
    out["arm_drop_range"] = [a_lo, a_hi]
    out["arm_spread"] = a_hi - a_lo
    out["seed_spread"] = hi - lo
    out["any_arm_removes_collapse"] = a_lo < 0.30

    print()
    print(f"Across four design choices the drop spans {a_lo:.4f} to {a_hi:.4f}, a spread")
    print(f"of {a_hi - a_lo:.4f}. Across six random seeds, changing nothing, it spans")
    print(f"{lo:.4f} to {hi:.4f}, a spread of {hi - lo:.4f}.")
    print()
    if out["any_arm_removes_collapse"]:
        print("An arm removed the collapse. The claim is architecture-specific and")
        print("the paper must be rewritten around that.")
    else:
        print("No design choice removes the collapse. Varying the loss, the head")
        print("width, or the encoder moves it about as much as varying the seed,")
        print("which is to say the collapse is a property of the evaluation and")
        print("not of this detector's construction.")
        print()
        twin = _twin_drops()
        if twin:
            d1, d2 = twin
            print("For contrast, the twin control changes what the training set")
            print(f"contains and nothing else: the drop goes {d1:.4f} -> {d2:.4f}, a")
            print(f"move of {d1 - d2:.4f}. Exposure moves the collapse "
                  f"{(d1 - d2) / (a_hi - a_lo):.0f}x further than")
            print("the widest gap between these four design choices.")
            out["twin_control"] = {"held_out_drop": d1, "exposed_drop": d2,
                                   "move": d1 - d2}

    dest = REPORTS / "e5_ablation.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
