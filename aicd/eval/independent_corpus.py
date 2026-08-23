"""Build an evaluation corpus whose provenance is independent of DroidCollection.

Every number in this work so far comes from one source collection, and a
reviewer is right to treat that as the study's most serious structural limit.
The remedy is not more of the same data but data assembled by other people, for
another purpose, from other generators.

AIGCodeSet (Demirok and Kutlu, SIU 2025) fits, and it fits for reasons worth
stating rather than assuming:

  independent assembly   built by an unrelated group in late 2024, with its own
                         prompting pipeline and its own quality decisions.

  an unseen generator    its LLM column holds CODESTRAL, LLAMA and GEMINI.
                         Gemini appears nowhere in DroidCollection's generator
                         list. Note that Gemma does appear there and is a
                         different model; the two must not be conflated.

  datable human code     its human submissions carry timestamps from July 2016
                         to September 2020, before Codex and before ChatGPT.
                         Their human authorship is a matter of record rather
                         than an assumption.

  a shared upstream      the human side is drawn from CodeNet, which
                         DroidCollection also draws on. That is a genuine
                         hazard, so this module measures the overlap instead of
                         waiting for a referee to ask about it, and can drop
                         every row that collides.

Label mapping. AIGCodeSet is binary, and our scheme has four classes:

  human                  label 0.
  machine                LLM rows generated from the problem description alone
                         (status Generate).
  hybrid                 LLM rows generated from the description *together with
                         human source that failed* (status Wrong or Runtime).
                         The model is rewriting human code, which is what the
                         hybrid class denotes.
  adversarial            absent. No part of AIGCodeSet asks a model to imitate
                         human style, so this corpus cannot exercise that
                         class, and the evaluation says so rather than
                         pretending otherwise.

    python -m aicd.eval.independent_corpus            # build and audit
    python -m aicd.eval.independent_corpus --keep-overlap
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import re

import pandas as pd

from aicd import config as C

SRC = os.path.join("artifacts", "data", "independent", "aigcodeset", "data")
OUT = os.path.join("artifacts", "data", "independent")
HUMAN, MACHINE, HYBRID, ADVERSARIAL = 0, 1, 2, 3


def norm(code: str) -> str:
    """Whitespace-insensitive form, so reformatting does not hide a duplicate.

    The same normalisation the evasion analysis uses, for the same reason: an
    overlap that survives a reformat is still an overlap.
    """
    s = str(code).replace("\r\n", "\n").replace("\t", "    ")
    s = "\n".join(l.rstrip() for l in s.split("\n"))
    return re.sub(r"\n{2,}", "\n", s).strip()


def digest(code: str) -> str:
    return hashlib.sha1(norm(code).encode("utf-8", "replace")).hexdigest()


def load_aigcodeset() -> pd.DataFrame:
    root = C.ROOT / SRC
    h = pd.read_csv(root / "human_selected_dataset.csv")
    m = pd.read_csv(root / "created_dataset_with_llms.csv")

    h = h[["code", "problem_id", "submission_id", "date"]].copy()
    h["label"] = HUMAN
    h["generator"] = "human"
    h["origin"] = "human_submission"

    status = m["status_in_folder"].astype(str).str.lower()
    m = m[["code", "problem_id", "submission_id", "LLM"]].assign(
        label=status.map(lambda s: MACHINE if s == "generate" else HYBRID),
        generator=m["LLM"].astype(str).str.lower(),
        origin=status.values)
    m["date"] = pd.NA

    df = pd.concat([h, m], ignore_index=True)
    df["code"] = df["code"].astype(str)
    df = df[df["code"].str.len() >= 40].reset_index(drop=True)
    df["language"] = "python"
    return df


def droid_digests() -> set:
    """Hashes of every DroidCollection row we hold, read in chunks.

    Only the digest is retained. Holding the code of 1.06M rows costs several
    gigabytes and is not needed to answer the question.
    """
    files = sorted(glob.glob(str(C.ROOT / "artifacts" / "data" / "hf" / "*.parquet")))
    if not files:
        raise SystemExit("DroidCollection parquets not found under artifacts/data/hf/")
    seen, total = set(), 0
    for f in files:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=20000, columns=["Code"]):
            col = batch.column("Code").to_pylist()
            total += len(col)
            seen.update(digest(c) for c in col if c)
        print(f"  hashed {os.path.basename(f)}  running total {total:,}")
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-overlap", action="store_true",
                    help="report the overlap but do not drop the rows")
    args = ap.parse_args()

    df = load_aigcodeset()
    print(f"AIGCodeSet: {len(df):,} rows after the 40-character floor")
    print("  by class   :", df["label"].value_counts().sort_index().to_dict())
    print("  by generator:", df["generator"].value_counts().to_dict())
    print()

    print("hashing DroidCollection to measure the overlap:")
    droid = droid_digests()
    print(f"  {len(droid):,} distinct normalised hashes\n")

    df["digest"] = df["code"].map(digest)
    df["in_droid"] = df["digest"].isin(droid)

    n_over = int(df["in_droid"].sum())
    print(f"rows also present in DroidCollection: {n_over:,} of {len(df):,} "
          f"({n_over/len(df):.2%})")
    if n_over:
        print("  by class:",
              df[df["in_droid"]]["label"].value_counts().sort_index().to_dict())

    # A row duplicated inside AIGCodeSet itself would inflate any metric.
    dupes = int(df["digest"].duplicated().sum())
    print(f"internal duplicates: {dupes:,}")

    kept = df if args.keep_overlap else df[~df["in_droid"]]
    kept = kept.drop_duplicates(subset="digest").reset_index(drop=True)
    print(f"\nretained: {len(kept):,} rows")
    print("  by class    :", kept["label"].value_counts().sort_index().to_dict())
    print("  by generator:", kept["generator"].value_counts().to_dict())

    os.makedirs(C.ROOT / OUT, exist_ok=True)
    dest = C.ROOT / OUT / "independent.parquet"
    kept[["code", "label", "generator", "language", "problem_id",
          "origin", "digest"]].to_parquet(dest, index=False)

    report = {
        "source": "basakdemirok/AIGCodeSet",
        "citation": "Demirok and Kutlu, SIU 2025, arXiv:2412.16594",
        "n_raw": int(len(df)),
        "n_overlapping_with_droidcollection": n_over,
        "overlap_fraction": float(n_over / len(df)),
        "internal_duplicates": dupes,
        "n_retained": int(len(kept)),
        "retained_by_class": {str(k): int(v) for k, v in
                              kept["label"].value_counts().sort_index().items()},
        "retained_by_generator": {str(k): int(v) for k, v in
                                  kept["generator"].value_counts().items()},
        "adversarial_class_present": False,
        "human_submission_dates": "2016-07 to 2020-09, before Codex and ChatGPT",
        "droid_hashes": len(droid),
    }
    rep = C.ROOT / "eval" / "reports" / "independent_corpus.json"
    io.open(rep, "w", encoding="utf-8").write(json.dumps(report, indent=2))
    print(f"\n-> {dest}")
    print(f"-> {rep}")


if __name__ == "__main__":
    main()
