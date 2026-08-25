"""Audit a third party's withheld categories against a third party's training split.

The audit of Section III chooses its own withheld categories, and a reader can
object that presence was guaranteed by that choice: the categories were drawn
from DroidCollection and then found in DroidCollection. The objection is fair
and the null distribution in the same section concedes the point.

This removes the choice from us. AICD Bench states which categories its Task 1
withholds, and DroidDetect-Base's model card states which split it trained on.
Neither list is ours. Asking whether the first occurs in the second is a
question about two published artefacts, and it is the question the paper's
argument actually needs answered.

AICD Bench Task 1 trains on Python, Java and C++ and evaluates on four splits of
increasing severity, the second and fourth of which withhold Golang, PHP, Rust,
JavaScript, C# and C. The third and fourth withhold research code and
general-purpose software as domains.

What this can and cannot settle is unchanged from the earlier audit: it
establishes presence in the split the card names, not survival of the
publisher's undisclosed filter.

    python -m aicd.eval.aicdbench_audit
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "aicd" / "eval" / "reports" / "aicdbench_audit.json"

# Stated by AICD Bench for Task 1, not chosen by us. Each entry is one
# category with the spellings DroidCollection might use for it, because the two
# corpora do not label C# or Go identically and a spelling mismatch would read
# as a category being absent when it is present.
AICD_HELD_OUT_LANGUAGES = {
    "Golang": ["go", "golang"],
    "PHP": ["php"],
    "Rust": ["rust"],
    "JavaScript": ["javascript", "js"],
    "C#": ["c#", "csharp", "c-sharp"],
    "C": ["c"],
}
AICD_TRAIN_LANGUAGES = ["python", "java", "cpp", "c++"]

# DroidCollection sources that are research code or general-purpose software
# rather than algorithmic problem solving. AICD Bench names the domains but not
# the source labels, so this mapping is ours and is flagged as such.
NON_ALGORITHMIC_SOURCES = ["arxiv", "thevault_function", "thevault_class",
                           "thevault_inline", "starcoder_data"]


def load_train() -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "aicd" / "artifacts" / "data" / "hf"
                                 / "train-*.parquet")))
    if not files:
        raise SystemExit("training shards not found under artifacts/data/hf/")
    parts = [pd.read_parquet(f, columns=["Language", "Source"]) for f in files]
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    df = load_train()
    n = len(df)
    lang = df["Language"].astype(str).str.strip().str.lower()
    src = df["Source"].astype(str).str.strip().str.lower()

    report = {
        "note": ("AICD Bench's Task 1 withheld categories, audited against the "
                 "DroidCollection training split DroidDetect-Base's card names. "
                 "Neither list was chosen by us. Presence in the named split is "
                 "not survival of the publisher's undisclosed filter."),
        "source": "AICD Bench (Orel et al., EACL 2026), Task 1",
        "train_rows": int(n),
        "languages": {},
        "domains": {},
    }

    present, absent, all_spellings = [], [], []
    for cat, spellings in AICD_HELD_OUT_LANGUAGES.items():
        all_spellings += spellings
        m = lang.isin(spellings)
        rows = int(m.sum())
        report["languages"][cat] = {
            "rows": rows, "share": round(rows / n, 6),
            "matched_as": sorted(set(lang[m])) if rows else [],
        }
        (present if rows else absent).append(cat)

    lang_mask = lang.isin(all_spellings)
    report["languages_union"] = {
        "rows": int(lang_mask.sum()),
        "share": round(float(lang_mask.mean()), 6),
        "categories_present": present,
        "categories_absent": absent,
    }

    for cat in NON_ALGORITHMIC_SOURCES:
        rows = int((src == cat).sum())
        report["domains"][cat] = {"rows": rows, "share": round(rows / n, 6)}
    dom_mask = src.isin(NON_ALGORITHMIC_SOURCES)
    report["domains_union"] = {"rows": int(dom_mask.sum()),
                               "share": round(float(dom_mask.mean()), 6)}

    both = lang_mask | dom_mask
    report["any_withheld_category"] = {"rows": int(both.sum()),
                                       "share": round(float(both.mean()), 6)}
    report["every_withheld_language_present"] = bool(
        not report["languages_union"]["categories_absent"])

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"  DroidDetect's named training split: {n:,} rows\n")
    print("  AICD Bench Task 1 withheld languages, in that split:")
    for cat, v in report["languages"].items():
        mark = "PRESENT" if v["rows"] else "ABSENT"
        seen = f"  as {v['matched_as']}" if v["matched_as"] else ""
        print(f"    {cat:<12} {v['rows']:>8,} rows  {v['share']*100:>6.2f}%  "
              f"{mark}{seen}")
    u = report["languages_union"]
    print(f"\n    union      {u['rows']:>8,} rows  {u['share']*100:>6.2f}%")
    print(f"    every withheld language present: "
          f"{report['every_withheld_language_present']}")
    d = report["domains_union"]
    print(f"\n  non-algorithmic sources (our mapping of their domains):")
    print(f"    union      {d['rows']:>8,} rows  {d['share']*100:>6.2f}%")
    a = report["any_withheld_category"]
    print(f"\n  any Task 1 withheld category: {a['rows']:,} rows "
          f"({a['share']*100:.2f}%)")
    print(f"  -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
