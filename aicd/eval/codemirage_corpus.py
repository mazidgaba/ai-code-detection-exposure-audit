"""E7: assemble CodeMirage as a second external corpus, and measure its overlap.

All three reviewers rejected AIGCodeSet as an independent check, and they were
right: it is Python only, it lacks a fourth class, and its human code descends
from Project CodeNet, the same upstream our own collection draws on. A second
corpus has to differ where that one does not.

CodeMirage (Guo et al., NeurIPS 2025) does. Its human half is drawn from
CodeParrot's Github-Code-Clean, sanitised in May 2022 before code LLMs were
widely deployed, so it is neither CodeNet nor plausibly contaminated. It spans
ten languages, two of which, Ruby and HTML, appear nowhere in DroidCollection.
And it carries paraphrased variants alongside plain generations, which is the
nearest thing available to the adversarial class AIGCodeSet has none of.

What makes it worth more than a second data point is that its generators split
into three tiers against our training exposure, and the middle tier is the one
this paper is actually about:

  unseen family        Anthropic, OpenAI o-series
  seen family, unseen model
                       Gemini 2.0, DeepSeek V3 and R1
  seen family and model
                       GPT-4o-mini, Llama-3.3-70B, Qwen2.5-Coder-32B

If accuracy falls across those tiers in order, the paper's thesis reproduces on
data neither we nor the DroidCollection authors built, as a gradient rather
than a binary.

    python -m aicd.eval.codemirage_corpus
    python -m aicd.eval.codemirage_corpus --split train --keep-overlap
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "aicd" / "artifacts" / "data" / "independent"
REPO = "HanxiGuo/CodeMirage"
HUMAN, MACHINE, HYBRID, ADVERSARIAL = 0, 1, 2, 3

# Each generator's family as DroidCollection would name it, and whether that
# family and that exact model occur in our training data. Derived by comparing
# CodeMirage's `source` values against the model_family and generator columns
# of the built corpus; see the module docstring for the three tiers.
GENERATORS = {
    "claude-3.5-haiku":              ("anthropic",       False, False),
    "o3-mini":                       ("openai-o-series", False, False),
    "gemini-2.0-flash":              ("google",          True,  False),
    "gemini-2.0-pro-exp":            ("google",          True,  False),
    "gemini-2.0-flash-thinking-exp": ("google",          True,  False),
    "deepseek-v3":                   ("deepseek-ai",     True,  False),
    "deepseek-r1":                   ("deepseek-ai",     True,  False),
    "gpt-4o-mini":                   ("gpt-4o-mini",     True,  True),
    "llama3.3-70b":                  ("meta-llama",      True,  True),
    "qwen2.5-coder":                 ("qwen",            True,  True),
}


# The one CodeMirage row that also occurs in DroidCollection, found by hashing
# all 1,058,248 rows of the latter against all 62,995 of the former. It is a
# human-written JavaScript file, which is the expected kind of coincidence:
# both corpora draw human JavaScript from public repositories. Recorded so a
# run that skips the audit still produces the audited corpus rather than one
# row larger.
KNOWN_OVERLAP = {"fa8aa624ad9e19d49850950a6a1ac7a42aa79215"}


def tier(family_seen: bool, model_seen: bool) -> str:
    if not family_seen:
        return "unseen_family"
    return "seen_family_seen_model" if model_seen else "seen_family_unseen_model"


def norm(code: str) -> str:
    """Whitespace-insensitive form, so reformatting cannot hide a duplicate.

    The same normalisation the AIGCodeSet audit uses, for the same reason and
    so the two overlap figures mean the same thing.
    """
    s = str(code).replace("\r\n", "\n").replace("\t", "    ")
    s = "\n".join(l.rstrip() for l in s.split("\n"))
    return re.sub(r"\n{2,}", "\n", s).strip()


def digest(code: str) -> str:
    return hashlib.sha1(norm(code).encode("utf-8", "replace")).hexdigest()


def droid_digests() -> set:
    """Hashes of every DroidCollection row, read in batches.

    Only the digest is kept: holding a million code strings costs gigabytes and
    answers nothing extra.
    """
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(ROOT / "aicd" / "artifacts" / "data" / "hf" / "*.parquet")))
    if not files:
        raise SystemExit("DroidCollection parquets not found under artifacts/data/hf/")
    seen, total = set(), 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=20000, columns=["Code"]):
            col = batch.column("Code").to_pylist()
            total += len(col)
            seen.update(digest(c) for c in col if c)
        print(f"  hashed {os.path.basename(f)}  running total {total:,}")
    return seen


def load(split: str) -> pd.DataFrame:
    from datasets import load_dataset
    print(f"downloading {REPO} [{split}] ...")
    ds = load_dataset(REPO, split=split)
    df = ds.to_pandas()
    need = {"code", "language", "source", "variant"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"CodeMirage schema changed; missing {sorted(missing)}")

    src = df["source"].astype(str)
    var = df["variant"].astype(str)

    # Human rows carry no variant. Plain generations are machine code. The
    # paraphrased ones are machine code deliberately rewritten to defeat a
    # detector, which is the closest thing this corpus has to our adversarial
    # class. It is not identical: ours is code prompted to imitate a human,
    # theirs is code transformed after the fact. The paper must say so rather
    # than let the mapping pass unremarked.
    label = pd.Series(MACHINE, index=df.index)
    label[src.str.lower() == "human"] = HUMAN
    label[var.str.lower() == "paraphrased"] = ADVERSARIAL

    out = pd.DataFrame({
        "code": df["code"].astype(str),
        "label": label.astype(int),
        "language": df["language"].astype(str).str.lower(),
        "generator": src.str.lower(),
        "variant": var,
    })
    fam, fseen, mseen = [], [], []
    for g in out["generator"]:
        f, a, b = GENERATORS.get(g, ("human" if g == "human" else "unknown",
                                     True, True))
        fam.append(f), fseen.append(a), mseen.append(b)
    out["model_family"] = fam
    out["exposure"] = [("human" if g == "human" else tier(a, b))
                       for g, a, b in zip(out["generator"], fseen, mseen)]

    unknown = sorted(set(out.loc[out["model_family"] == "unknown", "generator"]))
    if unknown:
        raise SystemExit(
            f"CodeMirage contains generators this module does not classify: "
            f"{unknown}. Classify them against DroidCollection's families "
            "before using the corpus, or the exposure tiers are wrong.")

    out = out[out["code"].str.len() >= 40].reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--keep-overlap", action="store_true",
                    help="report rows shared with DroidCollection but keep them")
    ap.add_argument("--skip-overlap", action="store_true",
                    help="do not hash DroidCollection. Only for a machine that "
                         "has already measured it: the audit is the point of "
                         "this module and skipping it quietly would defeat it")
    args = ap.parse_args()

    df = load(args.split)
    print(f"\nloaded {len(df):,} rows")
    print(df.groupby(["exposure"]).size().to_string())

    df["digest"] = [digest(c) for c in df["code"]]
    if args.skip_overlap:
        # Skipping the audit must not quietly produce a different corpus than
        # the audited one. The full hash found exactly one overlapping row, a
        # human JavaScript file, and its digest is recorded here so a run that
        # skips the audit still drops it and lands on the same row count.
        print("\nskipping the DroidCollection overlap audit as instructed. It "
              "was measured once at 1 row in 62,995; that row is dropped here "
              "by its recorded digest so this build matches the audited one.")
        dup = df["digest"].isin(KNOWN_OVERLAP)
        print(f"  dropped {int(dup.sum())} by recorded digest")
    else:
        print("\nhashing DroidCollection to measure overlap ...")
        seen = droid_digests()
        dup = df["digest"].isin(seen)
        print(f"\noverlap with DroidCollection: {int(dup.sum()):,} of "
              f"{len(df):,} ({dup.mean():.4%})")

    internal = int(df.duplicated("digest").sum())
    if not args.keep_overlap:
        df = df[~dup].drop_duplicates("digest").reset_index(drop=True)
    print(f"internal duplicates: {internal:,}")
    print(f"retained: {len(df):,}")

    os.makedirs(OUT, exist_ok=True)
    dest = OUT / f"codemirage_{args.split}.parquet"
    df.to_parquet(dest, index=False)

    report = {
        "split": args.split, "rows": int(len(df)),
        "overlap_with_droid": int(dup.sum()),
        "overlap_share": float(dup.mean()),
        "internal_duplicates": internal,
        "by_exposure": {k: int(v) for k, v in df.groupby("exposure").size().items()},
        "by_label": {k: int(v) for k, v in df.groupby("label").size().items()},
        "by_language": {k: int(v) for k, v in df.groupby("language").size().items()},
        "languages_absent_from_droid": sorted(
            set(df["language"]) - {"python", "java", "csharp", "cpp",
                                   "javascript", "go", "c", "php", "rust"}),
    }
    rep = ROOT / "aicd" / "eval" / "reports" / f"codemirage_{args.split}.json"
    os.makedirs(rep.parent, exist_ok=True)
    rep.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nlanguages DroidCollection does not contain at all: "
          f"{report['languages_absent_from_droid']}")
    print(f"written -> {dest}")
    print(f"written -> {rep}")


if __name__ == "__main__":
    main()
