"""Verify that the semantics-preserving rewrites preserve syntax.

The mechanism claim rests on rewrites that are asserted to preserve semantics
because each is constructed to. That is not evidence. A rewrite that silently
corrupted a fraction of its inputs would produce exactly the result the paper
reports: a large drop in accuracy on the altered files. Until the rewrites are
checked, 0.394 macro-F1 for renaming is indistinguishable from a corruption
rate of unknown size.

This checks the strongest property that can be checked cheaply and across every
language in the corpus: **the rewritten file still parses**. A file that parsed
before and does not parse after has been broken, whatever the rewrite intended.

Parsing is necessary for semantic preservation and not sufficient. A rename that
collided two distinct identifiers would still parse and would still change
behaviour, so a second check looks for exactly that: whether the identifier
renaming is injective on each file. Between them these two catch the failure
modes that would fake the paper's result.

What is not checked here is behaviour under execution, which would need a test
suite per row and is not available for this corpus. That limit is stated in the
manuscript rather than papered over.

    python -m aicd.eval.semantics_check
    python -m aicd.eval.semantics_check --n 4000 --transform rename_identifiers
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from aicd.models.transforms import IDENT, KEYWORDS, TRANSFORMS

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "aicd" / "eval" / "reports"

# tree-sitter names for the corpus languages.
GRAMMAR = {"python": "python", "java": "java", "csharp": "csharp",
           "cpp": "cpp", "javascript": "javascript", "go": "go",
           "c": "c", "php": "php", "rust": "rust"}


def parser_for(lang: str):
    from tree_sitter_language_pack import get_parser
    name = GRAMMAR.get(lang)
    if not name:
        return None
    try:
        return get_parser(name)
    except Exception:
        return None


def parses(parser, code: str) -> bool:
    """True if the source contains no ERROR node.

    tree-sitter is error-tolerant and always returns a tree, so the tree's own
    error flag is the signal rather than whether parsing raised.
    """
    try:
        tree = parser.parse(code.encode("utf-8", "surrogatepass"))
    except Exception:
        return False
    return not tree.root_node.has_error


def rename_is_injective(before: str, after: str) -> bool | None:
    """Whether the rename mapped distinct identifiers to distinct names.

    A collision would preserve syntax and change meaning, which is the failure
    a parse check cannot see. Returns None when the two files carry different
    identifier counts, where the question does not apply.
    """
    a = [t for t in IDENT.findall(before) if t not in KEYWORDS]
    b = [t for t in IDENT.findall(after) if t not in KEYWORDS]
    if len(a) != len(b):
        return None
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for x, y in zip(a, b):
        if mapping.setdefault(x, y) != y:
            return False
        if reverse.setdefault(y, x) != x:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="rows per language cap")
    ap.add_argument("--transform", default=None, help="check one rewrite only")
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    corpus = ROOT / "aicd" / "artifacts" / "data" / "splits.parquet"
    if not corpus.exists():
        raise SystemExit(f"{corpus} not found")
    df = pd.read_parquet(corpus, columns=["code", "language", "slice"])
    df = df[df["slice"] == "s1_in_distribution"]
    rng = np.random.default_rng(args.seed)
    if args.n and len(df) > args.n:
        df = df.iloc[rng.choice(len(df), args.n, replace=False)]

    names = [args.transform] if args.transform else list(TRANSFORMS)
    parsers = {lang: parser_for(lang) for lang in df["language"].unique()}
    usable = {k: v for k, v in parsers.items() if v is not None}
    print(f"{len(df):,} rows, {len(usable)} of {len(parsers)} languages parseable")
    missing = sorted(k for k, v in parsers.items() if v is None)
    if missing:
        print(f"  no grammar available for: {missing}")

    out = {"n_rows": int(len(df)), "condition": "s1_in_distribution",
           "languages_checked": sorted(usable), "transforms": {}}

    for name in names:
        fn = TRANSFORMS[name]
        broke, checked, altered = 0, 0, 0
        by_lang = Counter()
        noninjective = 0
        inj_checked = 0
        for code, lang in zip(df["code"], df["language"]):
            p = usable.get(lang)
            if p is None or not parses(p, code):
                continue                      # only files that parsed to begin with
            checked += 1
            new = fn(code, lang)
            if new == code:
                continue
            altered += 1
            if not parses(p, new):
                broke += 1
                by_lang[lang] += 1
            if name == "rename_identifiers":
                inj = rename_is_injective(code, new)
                if inj is not None:
                    inj_checked += 1
                    noninjective += (inj is False)

        rate = broke / altered if altered else 0.0
        rec = {"parsed_before": checked, "altered": altered, "broke_parse": broke,
               "break_rate": rate, "by_language": dict(by_lang)}
        if name == "rename_identifiers" and inj_checked:
            rec["injective_checked"] = inj_checked
            rec["non_injective"] = noninjective
            rec["collision_rate"] = noninjective / inj_checked
        out["transforms"][name] = rec
        extra = ""
        if "collision_rate" in rec:
            extra = f"   collisions {rec['collision_rate']:.4%}"
        print(f"  {name:20s} altered {altered:>6,}  broke {broke:>5,}  "
              f"({rate:.3%}){extra}")

    worst = max(out["transforms"].items(), key=lambda kv: kv[1]["break_rate"])
    out["max_break_rate"] = worst[1]["break_rate"]
    out["worst_transform"] = worst[0]
    print()
    print(f"Highest breakage: {worst[0]} at {worst[1]['break_rate']:.3%} of the "
          f"files it alters.")
    print()
    if worst[1]["break_rate"] < 0.01:
        print("Under 1%. The measured accuracy drops cannot be explained by")
        print("corruption: renaming costs 0.394 macro-F1 on the files it alters,")
        print("which is two orders of magnitude larger than the fraction it breaks.")
    else:
        print("Above 1%. A corruption of this size is large enough to contribute")
        print("to the measured effect, and the mechanism section must say so and")
        print("report the effect on the subset that still parses.")

    dest = REPORTS / "semantics_check.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
