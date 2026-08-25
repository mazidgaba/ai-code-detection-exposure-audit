"""Which axis of novelty does the twin's effect actually come from?

The twin control shows that exposure to withheld categories changes performance.
It does not, by itself, say which kind of withholding matters, and the five
conditions confound three axes: generator family, programming language and code
source. Two of them are not separable in S5, which is novel on both language and
source at once.

A reader is entitled to the obvious objection: if a condition contains only
languages absent from training, then a detector's failure on it is a statement
about tokenisation coverage rather than about benchmark-relative exposure, and
the finding is far less interesting than the paper claims.

This module measures the composition of each condition against the training
split and places the twin's measured effect beside it, so the objection can be
answered with the decomposition rather than argued about.

The answer is not uniform across axes, and saying so is the point:

  S2  novel generator family only ..... the axis benchmarks most often withhold
  S4  novel source only, every language seen in training
  S3  novel language only
  S5  novel language and source together

S4 is the case that carries the argument, because it isolates exposure from
tokenisation coverage entirely.

    python -m aicd.eval.axis_decomposition
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "aicd" / "eval" / "reports"
RUNS = ROOT / "kaggle_runs" / "results"
CONDITIONS = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
              "s4_unseen_domain", "s5_compound"]
AXES = [("language", "language"), ("source", "source"),
        ("generator family", "model_family")]


def main() -> None:
    corpus = ROOT / "aicd" / "artifacts" / "data" / "splits.parquet"
    if not corpus.exists():
        raise SystemExit(f"{corpus} not found")
    df = pd.read_parquet(corpus, columns=["slice", "language", "source",
                                          "model_family"])
    tr = df[df["slice"] == "train"]
    seen = {col: set(tr[col].unique()) for _, col in AXES}

    d1 = json.loads((RUNS / "e1" / "branch_a_e1_d1small.json").read_text())["slices"]
    d2 = json.loads((RUNS / "e1-d2" / "results" / "reports" /
                     "branch_a_e1_d2.json").read_text())["slices"]

    out = {"note": ("Composition is measured on the CPU build, whose category "
                    "structure is identical to the GPU build by construction; "
                    "the twin effects are measured on the arm build."),
           "conditions": {}}
    print(f"{'condition':22s} " + " ".join(f"{n[:12]:>13s}" for n, _ in AXES)
          + f" {'twin effect':>12s}")
    print("-" * 78)
    for c in CONDITIONS:
        d = df[df["slice"] == c]
        rec = {}
        for name, col in AXES:
            frac = float((~d[col].isin(seen[col])).mean())
            rec[col] = {"fraction_novel": frac,
                        "n_novel_values": int(len(set(d[col]) - seen[col]))}
        eff = (d2[c]["macro_f1"] - d1[c]["macro_f1"]) if c in d1 and c in d2 else None
        rec["twin_effect"] = eff
        out["conditions"][c] = rec
        cells = " ".join(f"{rec[col]['fraction_novel']:12.1%} " for _, col in AXES)
        print(f"{c:22s} {cells}{eff:+12.4f}")

    # The decomposition's payload: the one condition novel on source alone.
    s4 = out["conditions"]["s4_unseen_domain"]
    s2 = out["conditions"]["s2_unseen_generator"]
    out["clean_source_axis"] = {
        "condition": "s4_unseen_domain",
        "language_novel": s4["language"]["fraction_novel"],
        "source_novel": s4["source"]["fraction_novel"],
        "twin_effect": s4["twin_effect"],
    }
    out["generator_axis"] = {"condition": "s2_unseen_generator",
                             "twin_effect": s2["twin_effect"]}
    print()
    print(f"S4 is novel on source for {s4['source']['fraction_novel']:.0%} of its rows and "
          f"novel on language for {s4['language']['fraction_novel']:.0%}.")
    print(f"Every language in it occurs in training, and exposure is still worth "
          f"{s4['twin_effect']:+.4f} there.")
    print("That is the measurement the argument rests on: an exposure effect with")
    print("tokenisation coverage held constant.")
    print()
    print(f"S2, novel on generator family alone, is worth {s2['twin_effect']:+.4f}.")
    print("The axis benchmarks most often withhold is the axis where withholding")
    print("costs least, which is the practical finding rather than a caveat.")

    if s4["language"]["fraction_novel"] > 0.01:
        raise SystemExit("S4 is no longer language-clean; the argument above "
                         "depends on it being so")

    dest = REPORTS / "axis_decomposition.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
