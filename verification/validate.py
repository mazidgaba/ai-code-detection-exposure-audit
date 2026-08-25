"""Mechanical sanity checks on the LaTeX source.

No local TeX distribution is available, so this catches the classes of error
that would otherwise only surface at compile time: unbalanced environments or
braces, dangling references, uncited or undefined bibliography entries, missing
figure files, and rows whose cell count disagrees with the column spec.
"""
from __future__ import annotations

import collections
import os
import re
import sys

BS = chr(92)  # backslash, kept out of the regex literals for clarity


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    tex = os.path.join(here, "main.tex")
    if not os.path.exists(tex):
        # The public artifact ships the verification scripts but not the
        # manuscript, which is unpublished. Explain that rather than dying on a
        # traceback that reads like a broken checkout.
        print("main.tex is not present, so there is nothing to validate.\n"
              "This script checks the LaTeX source for unbalanced environments, "
              "dangling\nreferences, uncited bibliography entries and malformed "
              "table rows. It is\npublished alongside review.py to document what "
              "was checked, and runs once the\nmanuscript is available.")
        return 0
    src = open(tex, encoding="utf-8").read()
    ok = True

    def fail(msg):
        nonlocal ok
        ok = False
        print("  FAIL:", msg)

    # --- environments -------------------------------------------------------
    begins = re.findall(BS + BS + r"begin\{(\w+\*?)\}", src)
    ends = re.findall(BS + BS + r"end\{(\w+\*?)\}", src)
    cb, ce = collections.Counter(begins), collections.Counter(ends)
    for k in sorted(set(cb) | set(ce)):
        if cb[k] != ce[k]:
            fail(f"env {k}: {cb[k]} begin vs {ce[k]} end")
    print(f"environments : {len(begins)} begin / {len(ends)} end")

    # --- braces -------------------------------------------------------------
    depth, line, first_bad = 0, 1, None
    for i, ch in enumerate(src):
        if ch == "\n":
            line += 1
        elif ch == "{" and (i == 0 or src[i - 1] != BS):
            depth += 1
        elif ch == "}" and (i == 0 or src[i - 1] != BS):
            depth -= 1
            if depth < 0 and first_bad is None:
                first_bad = line
    if first_bad:
        fail(f"unbalanced closing brace near line {first_bad}")
    elif depth != 0:
        fail(f"brace depth ends at {depth}")
    else:
        print("braces       : balanced")

    # --- labels and refs ----------------------------------------------------
    labels = set(re.findall(BS + BS + r"label\{([^}]+)\}", src))
    refs = set(re.findall(BS + BS + r"ref\{([^}]+)\}", src))
    if refs - labels:
        fail(f"refs with no label: {sorted(refs - labels)}")
    print(f"labels/refs  : {len(labels)} labels, {len(refs)} refs, "
          f"unreferenced: {sorted(labels - refs) or 'none'}")

    # --- citations ----------------------------------------------------------
    cites: set[str] = set()
    for c in re.findall(BS + BS + r"cite\{([^}]+)\}", src):
        cites |= {x.strip() for x in c.split(",")}
    bibs = set(re.findall(BS + BS + r"bibitem\{([^}]+)\}", src))
    if cites - bibs:
        fail(f"cited but undefined: {sorted(cites - bibs)}")
    print(f"citations    : {len(cites)} cited, {len(bibs)} bibitems, "
          f"uncited: {sorted(bibs - cites) or 'none'}")

    # --- figures ------------------------------------------------------------
    figs = re.findall(BS + BS + r"includegraphics\[[^\]]*\]\{([^}]+)\}", src)
    for f in figs:
        p = os.path.join(here, f)
        alt = os.path.join(here, "..", "aicd", "docs", f)
        exists = os.path.exists(p) or os.path.exists(alt)
        print(f"  {'ok     ' if exists else 'MISSING'} {f}")
        if not exists:
            fail(f"figure not found: {f}")

    # --- shell-mangled commands ---------------------------------------------
    # Editing TeX through a shell can silently eat the backslash of a control
    # word whose first letter is an escape character, turning "\ref{" into
    # "ef{" and "\newline" into a literal newline. These survive to the PDF as
    # stray text, so scan for command-like words that lost their backslash.
    STUBS = ["ef", "ite", "abel", "ewcommand", "ho", "extbf", "extit", "aggedright"]
    hits = []
    for stub in STUBS:
        # BS must be doubled inside the class or it escapes the closing "]".
        for m in re.finditer(r"(?<![A-Za-z" + BS + BS + r"])" + stub + r"\{", src):
            ln = src[: m.start()].count("\n") + 1
            ctx = src[max(0, m.start() - 40): m.start() + 22].replace("\n", " ")
            hits.append((ln, stub, ctx))
    for ln, stub, ctx in sorted(hits):
        fail(f"line {ln}: '{stub}{{' looks like a mangled command -- ...{ctx}...")
    if not hits:
        print("commands     : no mangled control words")

    # --- em dashes ----------------------------------------------------------
    # House style bans the em dash. Both the TeX ligature "---" and the literal
    # U+2014 character are rejected; "--" (en dash, for numeric ranges) is fine.
    em_tex = [src[: m.start()].count("\n") + 1
              for m in re.finditer(r"(?<!-)---(?!-)", src)]
    em_uni = [src[: m.start()].count("\n") + 1 for m in re.finditer("—", src)]
    for ln in em_tex:
        fail(f"line {ln}: TeX em dash '---'")
    for ln in em_uni:
        fail(f"line {ln}: literal em dash U+2014")
    if not em_tex and not em_uni:
        print("em dashes    : none")

    # --- table geometry -----------------------------------------------------
    # The column spec may itself contain braces (e.g. "@{}lrrr@{}"), so match
    # balanced one-level nesting rather than stopping at the first "}".
    pat = (BS + BS + r"begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)"
           + BS + BS + r"end\{tabular\}")
    for i, (spec, body) in enumerate(re.findall(pat, src, re.S), 1):
        ncol = len(re.sub(r"[^lcr]", "", spec))
        for raw in body.split(BS * 2):
            row = raw.strip()
            if not row or any(r in row for r in ("toprule", "midrule", "bottomrule")):
                continue
            row = re.sub(BS + BS + r"multicolumn\{(\d+)\}", lambda m: "&" * (int(m.group(1)) - 1), row)
            n = row.count("&") + 1
            if n != ncol:
                print(f"  [warn] table {i}: {n} cells vs spec {ncol}: {row[:58]}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
