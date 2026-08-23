"""Verify, rather than infer, what the published detector was exposed to.

The paper's central argument is that our five conditions are not out of
distribution for DroidDetect-Base, because the categories we withhold are
present in the data it trained on. Until now that was an inference from the
model card, and a reviewer is entitled to reject an inference standing where a
measurement belongs.

It does not have to stay an inference. DroidDetect-Base's card states it was
trained on the filtered training set of DroidCollection, and the training
shards of DroidCollection are public. Reading the category columns of those
shards establishes directly whether each withheld family, language and source
occurs in them, and in what quantity.

What this can and cannot settle. It settles whether the categories are present
in the training split the model card names. It cannot settle whether the
publisher's own filtering removed them afterwards, since that filter is not
released. The distinction is recorded in the output and in the paper.

    python -m aicd.eval.exposure_audit
"""
from __future__ import annotations

import glob
import io
import json
import os

import pandas as pd

from aicd import config as C

# The categories the corpus withholds from Branch A, as named in the paper.
WITHHELD = {
    "generator family": ["microsoft", "mistralai"],
    "language": ["go", "javascript"],
    "source": ["thevault_class", "thevault_inline", "arxiv"],
}
COLUMN = {"generator family": "Model_Family", "language": "Language",
          "source": "Source"}


def load_train_categories() -> pd.DataFrame:
    """Read only the category columns; the code column is large and unused."""
    files = sorted(glob.glob(str(C.ROOT / "artifacts" / "data" / "hf" /
                                 "train-*.parquet")))
    if not files:
        raise SystemExit(
            "DroidCollection training shards not found under "
            "artifacts/data/hf/. Run the download step first.")
    cols = ["Model_Family", "Language", "Source", "Label"]
    parts = [pd.read_parquet(f, columns=cols) for f in files]
    df = pd.concat(parts, ignore_index=True)
    print(f"read {len(files)} training shards, {len(df):,} rows")
    return df


def main() -> None:
    df = load_train_categories()
    norm = {c: df[c].astype(str).str.strip().str.lower() for c in COLUMN.values()}

    out = {"n_train_rows": int(len(df)), "shards": 3, "axes": {}}
    print()
    print(f"{'axis':17s} {'withheld category':20s} {'rows in train':>14s} {'share':>8s}")
    print("-" * 63)

    all_present = True
    for axis, cats in WITHHELD.items():
        col = norm[COLUMN[axis]]
        rec = {}
        for cat in cats:
            n = int((col == cat).sum())
            rec[cat] = {"rows": n, "share": round(n / len(df), 6)}
            if n == 0:
                all_present = False
            print(f"{axis:17s} {cat:20s} {n:14,d} {n/len(df):7.3%}")
        out["axes"][axis] = rec

    # Record the full vocabulary so a reader can check our category names match
    # the ones the dataset actually uses, rather than trusting our spelling.
    out["observed_values"] = {
        c: sorted(norm[c].value_counts().index.tolist())[:60]
        for c in COLUMN.values()
    }
    out["all_withheld_categories_present_in_training_split"] = all_present

    # A row can be Go *and* Microsoft, so summing the rows above double-counts.
    # Report the union: rows touching at least one withheld category.
    mask = pd.Series(False, index=df.index)
    for axis, cats in WITHHELD.items():
        mask |= norm[COLUMN[axis]].isin(cats)
    union = int(mask.sum())
    naive = sum(v["rows"] for a in out["axes"].values() for v in a.values())
    out["union_rows_touching_any_withheld_category"] = union
    out["union_share"] = round(union / len(df), 6)
    out["naive_sum_double_counts"] = naive

    print("-" * 63)
    print(f"{'':17s} {'union (deduplicated)':20s} {union:14,d} {union/len(df):7.3%}")
    print(f"{'':17s} {'(naive sum)':20s} {naive:14,d} {naive/len(df):7.3%}")
    print()
    print("all withheld categories present in the training split:",
          "YES" if all_present else "NO")
    print("\nCaveat recorded in the report: the model card states a *filtered*")
    print("training set. The filter is not released, so this establishes")
    print("presence in the named split, not survival of that filter.")

    dest = C.ROOT / "eval" / "reports" / "exposure_audit.json"
    os.makedirs(dest.parent, exist_ok=True)
    io.open(dest, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
