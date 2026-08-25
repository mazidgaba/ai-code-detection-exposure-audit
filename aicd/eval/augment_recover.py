"""Recover E8, the augmentation mitigation, from its kernel log.

E6 establishes that the detector is sensitive to semantics-preserving rewrites.
The obvious remedy is to train through them, so that the surface cue stops being
predictive and the model is pushed toward something deeper. E8 does exactly
that: renaming and whitespace normalisation applied to half the training rows.

If it worked, the paper's finding would have a fix and the story would be about
brittle preprocessing. It does not work, and that matters more than a fix would:
it means the rewrite sensitivity of E6 is a symptom rather than the mechanism.

The comparison is against the six-seed spread, not against the single reference
run. The reference drop of 0.6599 is the largest of the six seed drops, which
range from 0.6159 to 0.6599; measuring a mitigation against the maximum of a
noisy baseline makes any arm look like an improvement. An arm has done something
only if it lands outside the range the recipe already produces on its own.

    python -m aicd.eval.augment_recover
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "aicd" / "eval" / "reports"
LOG = ROOT / "kaggle_runs" / "results" / "e8_augment_kernel.log"
DEST = REPORTS / "e8_augment.json"

CONDITIONS = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
              "s4_unseen_domain", "s5_compound"]
ROW = re.compile(
    r"^(s[1-5]_[a-z_]+)\s+([\d,]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
    re.M)
DROP = re.compile(r"S1 -> S5:\s*[\d.]+ -> [\d.]+\s*\(drop ([\d.]+)\)")
AUG = re.compile(r"augmentation: (\[[^\]]*\]), target (\d+)% of ([\d,]+) rows")
ALT = re.compile(r"([\d,]+) selected, ([\d,]+) actually altered \(([\d.]+)% of those selected\)")


def _text(path: Path) -> str:
    raw = io.open(path, encoding="utf-8", errors="ignore").read()
    try:
        rows = json.loads(raw)
        if isinstance(rows, list):
            return "\n".join(r.get("data", "") for r in rows if isinstance(r, dict))
    except Exception:
        pass
    return raw


def main() -> None:
    if not LOG.exists():
        raise SystemExit(f"{LOG} not found")
    t = _text(LOG)

    by = {}
    for m in ROW.finditer(t):
        cond, f1 = m.group(1), float(m.group(3))
        if cond in by and abs(by[cond]["macro_f1"] - f1) > 1e-9:
            raise SystemExit(f"conflicting macro-F1 for {cond} in the log")
        by[cond] = {"n": int(m.group(2).replace(",", "")), "macro_f1": f1,
                    "weighted_f1": float(m.group(4)), "accuracy": float(m.group(5)),
                    "binary_auc": float(m.group(6)), "human_fpr": float(m.group(7))}
    missing = [c for c in CONDITIONS if c not in by]
    if missing:
        raise SystemExit(f"log has no macro-F1 for {missing}")

    d = DROP.findall(t)
    if not d:
        raise SystemExit("log has no drop summary line")
    drop = float(d[-1])
    table_drop = by["s1_in_distribution"]["macro_f1"] - by["s5_compound"]["macro_f1"]
    if abs(drop - table_drop) > 2e-4:
        raise SystemExit(f"drop line {drop} contradicts the table's {table_drop:.4f}")

    aug, alt = AUG.search(t), ALT.search(t)
    if not (aug and alt):
        raise SystemExit("log does not record what the augmentation did")

    seeds = json.loads((REPORTS / "seed_sweep_six.json").read_text(encoding="utf-8"))
    ref = {k: v for k, v in zip(CONDITIONS, seeds["seeds"]["20260818"])}
    drops = {k: v[0] - v[-1] for k, v in seeds["seeds"].items()}
    lo, hi = min(drops.values()), max(drops.values())

    out = {
        "model": "branch_a_e8_augment",
        "source": str(LOG.relative_to(ROOT)).replace("\\", "/"),
        "provenance": "kernel log",
        "augmentation": {
            "transforms": json.loads(aug.group(1).replace("'", '"')),
            "target_fraction": int(aug.group(2)) / 100,
            "train_rows": int(aug.group(3).replace(",", "")),
            "selected": int(alt.group(1).replace(",", "")),
            "actually_altered": int(alt.group(2).replace(",", "")),
            "altered_fraction_of_selected": float(alt.group(3)) / 100,
        },
        "slices": by, "drop": drop,
        "reference": {"by_condition": ref, "drop_min": lo, "drop_max": hi,
                      "drop_mean": seeds["drop_mean"]},
    }
    out["vs_seed_range"] = ("inside" if lo <= drop <= hi
                            else "below" if drop < lo else "above")

    a = out["augmentation"]
    print(f"augmented {a['actually_altered']:,} of {a['selected']:,} selected rows "
          f"({a['altered_fraction_of_selected']:.1%}) out of {a['train_rows']:,}")
    print(f"transforms: {', '.join(a['transforms'])}")
    print(f"\n{'condition':24s} {'augmented':>10s} {'reference':>10s} {'delta':>8s}")
    print("-" * 56)
    for c in CONDITIONS:
        delta = by[c]["macro_f1"] - ref[c]
        out["slices"][c]["delta_vs_reference"] = delta
        print(f"{c:24s} {by[c]['macro_f1']:10.4f} {ref[c]:10.4f} {delta:+8.4f}")
    print("-" * 56)
    print(f"{'S1 to S5 drop':24s} {drop:10.4f} {hi:10.4f}")
    print(f"\nsix-seed drop range {lo:.4f} to {hi:.4f}; this arm is {out['vs_seed_range']}.")
    print()
    if out["vs_seed_range"] == "inside":
        print("Augmentation does not measurably change the collapse: its drop falls")
        print("inside the range the same recipe produces from seed alone. Training")
        print("through the rewrites the detector is sensitive to therefore does not")
        print("restore transfer, so that sensitivity is a symptom and not the")
        print("mechanism.")
    else:
        print("The augmented arm falls outside the seed range. Report the direction")
        print("plainly and revisit what the paper says about mitigation.")

    worse = [(c, out["slices"][c]["delta_vs_reference"]) for c in CONDITIONS
             if out["slices"][c]["delta_vs_reference"] < 0]
    if worse:
        w = min(worse, key=lambda x: x[1])
        print(f"\nIt is also not free: {len(worse)} of five conditions are worse, "
              f"the largest being\n{w[0]} at {w[1]:+.4f}.")

    DEST.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
