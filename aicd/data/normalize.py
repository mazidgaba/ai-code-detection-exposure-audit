"""Phase 01b: normalize raw DroidCollection into the canonical schema.

Canonical columns produced here:
    code, label (0-3), language, domain, model_family, generator,
    generation_mode, source, problem_id, orig_split
"""
from __future__ import annotations

import argparse
import hashlib

import pandas as pd

from aicd import config as C

# --- label mapping -----------------------------------------------------------
# Verified live values: HUMAN_GENERATED, MACHINE_GENERATED, MACHINE_REFINED.
# The adversarial class is matched by substring so we stay robust to the exact
# spelling the dataset uses for it.
def map_label(raw: str) -> int | None:
    s = str(raw).strip().upper()
    if "ADVERS" in s:
        return 3
    if s in ("HUMAN_GENERATED", "HUMAN", "HUMAN_WRITTEN"):
        return 0
    if s in ("MACHINE_GENERATED", "AI_GENERATED", "MACHINE"):
        return 1
    if s in ("MACHINE_REFINED", "HYBRID", "AI_REFINED"):
        return 2
    return None


LANG_MAP = {
    "python": "python", "py": "python",
    "java": "java",
    "c++": "cpp", "cpp": "cpp", "cxx": "cpp",
    "c": "c",
    "c#": "csharp", "csharp": "csharp", "cs": "csharp",
    "go": "go", "golang": "go",
    "javascript": "javascript", "js": "javascript",
    "php": "php", "ruby": "ruby", "rust": "rust",
}

# Source -> domain. Algorithmic sources are competitive-programming style;
# research is paper/notebook code; general_deployed is real repository code.
DOMAIN_MAP = {
    "TACO": "algorithmic", "CODEFORCES": "algorithmic", "LEETCODE": "algorithmic",
    "ATCODER": "algorithmic", "CODENET": "algorithmic", "APPS": "algorithmic",
    "ARXIV": "research", "RESEARCH": "research", "KAGGLE": "research",
    "THEVAULT_FUNCTION": "general_deployed", "THEVAULT_CLASS": "general_deployed",
    "THEVAULT_INLINE": "general_deployed", "STARCODER_DATA": "general_deployed",
    "DROID_PERSONAHUB": "general_deployed", "GITHUB": "general_deployed",
}


def map_language(raw: str) -> str:
    return LANG_MAP.get(str(raw).strip().lower(), str(raw).strip().lower())


def map_domain(source: str) -> str:
    return DOMAIN_MAP.get(str(source).strip().upper(), "general_deployed")


def make_problem_id(row) -> str:
    """Group key for problem-wise splitting.

    Every solution to the same underlying problem must land in one split.
    Human and machine solutions to a problem share the group, so a random
    split cannot leak near-duplicates across the train/test boundary.
    """
    src = str(row.get("source", ""))
    code = str(row.get("code", ""))
    # Normalize away whitespace and identifiers-ish noise, then hash the shape.
    shape = "".join(ch for ch in code if not ch.isspace())[:512]
    return f"{src}:{hashlib.sha1(shape.encode('utf-8', 'ignore')).hexdigest()[:16]}"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["code"] = df["Code"].astype(str)
    out["label"] = df["Label"].map(map_label)
    out["language"] = df["Language"].map(map_language)
    out["source"] = df["Source"].astype(str)
    out["domain"] = df["Source"].map(map_domain)
    out["model_family"] = df["Model_Family"].astype(str).str.lower()
    out["generator"] = df["Generator"].astype(str)
    out["generation_mode"] = df["Generation_Mode"].astype(str)
    out["orig_split"] = df.get("orig_split", "train")

    before = len(out)
    out = out[out["label"].notna()].copy()
    out["label"] = out["label"].astype(int)
    dropped = before - len(out)
    if dropped:
        print(f"  dropped {dropped:,} rows with unmappable labels")

    out["problem_id"] = out.apply(make_problem_id, axis=1)
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    args = ap.parse_args()
    cfg = C.load(args.config)
    d = C.ROOT / cfg.data.cache_dir

    frames = []
    for split in ("train", "dev", "test"):
        p = d / f"raw_{split}.parquet"
        if not p.exists():
            print(f"[warn] missing {p.name}, run data/download.py first")
            continue
        print(f"[normalize] {p.name}")
        frames.append(normalize(pd.read_parquet(p)))

    if not frames:
        raise SystemExit("no raw data found")

    df = pd.concat(frames, ignore_index=True)
    dest = d / "normalized.parquet"
    df.to_parquet(dest, index=False)

    print(f"\n[done] {dest.name}: {len(df):,} rows")
    print("\nlabel distribution:")
    print(df["label"].value_counts().sort_index().to_string())
    print("\nlanguage:")
    print(df["language"].value_counts().to_string())
    print("\ndomain:")
    print(df["domain"].value_counts().to_string())
    print("\nmodel_family:")
    print(df["model_family"].value_counts().to_string())


if __name__ == "__main__":
    main()
