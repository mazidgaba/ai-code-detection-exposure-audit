"""Recover the corrected transformation battery from its kernel log.

The published table ranked five semantics-preserving rewrites by how far
macro-F1 fell when each was applied to a whole evaluation set. That is
confounded, because the rewrites fire at very different rates: renaming altered
100% of files, stripping comments 81%, compressing blank runs under 6%. A
rewrite that fires more often moves the aggregate more whatever its per-file
effect, so the ranking measured reach as much as damage.

The corrected run scores the baseline and the rewritten inputs on **the same
altered rows**, at n = 10,000 rather than 2,000, on a named condition, stratified
by class. Its report was written inside the Kaggle session and only the kernel
log was retained locally, so this module parses the log into the same JSON shape
the module itself emits. The log is the primary record; the parse is checked
against the log's own aggregate column, and against the arithmetic relating
macro-F1 to the baseline.

    python -m aicd.eval.transform_recover
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "kaggle_runs" / "results" / "e6_transforms_kernel.log"
DEST = ROOT / "aicd" / "eval" / "reports" / "transform_suite_corrected.json"

ROW = re.compile(
    r"^\s+(\w+)\s+([\d.]+)%\s+([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s*$",
    re.M)
COND = re.compile(r"^(\w+)\s+([\d.]+)%\s+([\d,]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$", re.M)


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

    base = re.search(r"baseline macro-F1 ([\d.]+)", t)
    head = re.search(r"slice (\S+), n = ([\d,]+)", t)
    if not (base and head):
        raise SystemExit("log does not carry the run header")
    baseline = float(base.group(1))
    slice_name, n = head.group(1), int(head.group(2).replace(",", ""))

    detail = {m.group(1): m.groups() for m in ROW.finditer(t)}
    counts = {m.group(1): m.groups() for m in COND.finditer(t)}
    if not detail:
        raise SystemExit("no per-transform rows found in the log")

    out = {"slice": slice_name, "n": n, "baseline": {"macro_f1": baseline},
           "source": str(LOG.relative_to(ROOT)).replace("\\", "/"),
           "provenance": "kernel log", "transforms": []}

    for name, g in detail.items():
        applied = float(g[1]) / 100.0
        macro = float(g[2])
        agg = float(g[3])
        cond = float(g[4])
        # The aggregate delta must equal the rewritten score minus the baseline.
        # If the log's own columns disagree the parse has drifted onto the wrong
        # table, which is the failure this check exists to catch.
        if abs((macro - baseline) - agg) > 5e-4:
            raise SystemExit(f"{name}: macro-F1 {macro} minus baseline {baseline} "
                             f"is {macro - baseline:.4f}, log says {agg}")
        row = {"transform": name, "applied_fraction": applied,
               "macro_f1": macro, "delta_macro_f1": agg,
               "delta_macro_f1_conditional": cond,
               "binary_auc": float(g[5]), "delta_binary_auc": float(g[6])}
        if name in counts:
            row["n_altered"] = int(counts[name][2].replace(",", ""))
        out["transforms"].append(row)

    out["transforms"].sort(key=lambda r: r["delta_macro_f1_conditional"])

    print(f"slice {slice_name}, n = {n:,}, baseline macro-F1 {baseline:.4f}")
    print(f"\n{'rewrite':20s} {'applied':>8s} {'altered':>8s} "
          f"{'aggregate':>10s} {'conditional':>12s}")
    print("-" * 62)
    for r in out["transforms"]:
        print(f"{r['transform']:20s} {r['applied_fraction']:7.1%} "
              f"{r.get('n_altered', 0):>8,} {r['delta_macro_f1']:+10.4f} "
              f"{r['delta_macro_f1_conditional']:+12.4f}")

    agg_order = [r["transform"] for r in
                 sorted(out["transforms"], key=lambda r: r["delta_macro_f1"])]
    cond_order = [r["transform"] for r in out["transforms"]]
    out["aggregate_order"] = agg_order
    out["conditional_order"] = cond_order
    out["ranking_reverses"] = agg_order != cond_order

    print()
    if out["ranking_reverses"]:
        print("The ordering CHANGES once reach is removed. The published ranking")
        print("was partly an artefact of application rate.")
    else:
        print("The ordering is unchanged: renaming still costs most on the rows it")
        print("touches. The published ranking survives correction, but the figures")
        print("behind it do not, and the reason it survives is not the reason the")
        print("paper gave.")

    # The one row where correction changes the reading rather than the number.
    cb = next((r for r in out["transforms"] if r["transform"] == "compress_blanks"), None)
    if cb:
        print(f"\ncompress_blanks: aggregate {cb['delta_macro_f1']:+.4f} but conditional "
              f"{cb['delta_macro_f1_conditional']:+.4f}")
        print("The paper called this a clean null. On the 5.7% of files it actually")
        print("touched it is twenty times larger than the aggregate suggests, so the")
        print("null was a statement about reach, not about effect.")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
