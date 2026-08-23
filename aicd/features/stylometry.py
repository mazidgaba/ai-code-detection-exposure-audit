"""Stylometric features.

The first seven are the Oedingen et al. (MDPI AI 2024) white-box set, which
reached 88.5% accuracy on unformatted Python by themselves. The rest add the
CoDet-M4 structural features -- empty-line count and function density were
among the strongest SHAP contributors there, because models space their code
out more consistently than people do.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

PUNCT_RE = re.compile(r"[^\w\s]")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]{1,}")
FUNC_RE = re.compile(r"\b(def|func|function|fn)\b|\w+\s*\([^)]*\)\s*\{")
BRANCH_RE = re.compile(r"\b(if|elif|else|for|while|case|switch|catch|except|&&|\|\||\?)\b")
COMMENT_RE = re.compile(r"(#.*?$)|(//.*?$)|(/\*.*?\*/)|(\"\"\".*?\"\"\")|('''.*?''')", re.S | re.M)

KEYWORDS = {
    "if", "else", "for", "while", "return", "def", "class", "import", "from",
    "try", "except", "catch", "throw", "new", "public", "private", "static",
    "void", "int", "const", "let", "var", "func", "package", "struct",
}


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def stylometry(code: str) -> dict[str, float]:
    lines = code.split("\n")
    n_lines = max(len(lines), 1)
    n_chars = max(len(code), 1)

    # --- Oedingen et al. white-box seven ---
    n_leading_ws = sum(len(l) - len(l.lstrip()) for l in lines)
    n_empty = sum(1 for l in lines if not l.strip())
    n_inline_ws = sum(l.strip().count(" ") for l in lines)
    n_punct = len(PUNCT_RE.findall(code))
    max_line = max((len(l) for l in lines), default=0)
    n_trailing_ws = sum(len(l) - len(l.rstrip()) for l in lines)
    n_indented = sum(1 for l in lines if l[:1] in (" ", "\t"))

    idents = IDENT_RE.findall(code)
    ident_lens = [len(i) for i in idents if i not in KEYWORDS] or [0]
    comments = COMMENT_RE.findall(code)
    comment_chars = sum(len("".join(g)) for g in comments)

    line_lens = [len(l) for l in lines]
    n_funcs = len(FUNC_RE.findall(code))
    branches = len(BRANCH_RE.findall(code))

    tabs = code.count("\t")
    spaces_indent = sum(1 for l in lines if l.startswith(" "))

    return {
        # the seven
        "n_leading_ws": n_leading_ws,
        "n_empty_lines": n_empty,
        "n_inline_ws": n_inline_ws,
        "n_punct": n_punct,
        "max_line_len": max_line,
        "n_trailing_ws": n_trailing_ws,
        "n_indented_lines": n_indented,
        # normalized variants -- raw counts scale with file size, ratios don't
        "r_empty_lines": n_empty / n_lines,
        "r_indented_lines": n_indented / n_lines,
        "r_trailing_ws": n_trailing_ws / n_chars,
        "r_punct": n_punct / n_chars,
        # size
        "n_lines": n_lines,
        "n_chars": n_chars,
        "mean_line_len": float(np.mean(line_lens)) if line_lens else 0.0,
        "std_line_len": float(np.std(line_lens)) if line_lens else 0.0,
        # identifiers
        "mean_ident_len": float(np.mean(ident_lens)),
        "std_ident_len": float(np.std(ident_lens)),
        "n_unique_idents": len(set(idents)),
        "r_unique_idents": len(set(idents)) / max(len(idents), 1),
        # comments
        "comment_density": comment_chars / n_chars,
        "n_comment_blocks": len(comments),
        # structure
        "func_density": n_funcs / n_lines,
        "mean_func_len": n_lines / max(n_funcs, 1),
        "cyclomatic": branches + 1,
        "cyclomatic_per_line": (branches + 1) / n_lines,
        # indentation style -- generators are far more consistent here
        "tab_count": tabs,
        "r_space_indent": spaces_indent / n_lines,
        "indent_is_mixed": 1.0 if (tabs > 0 and spaces_indent > 0) else 0.0,
        # information density
        "char_entropy": _entropy(code),
        "r_whitespace": sum(c.isspace() for c in code) / n_chars,
        "r_upper": sum(c.isupper() for c in code) / n_chars,
        "r_digit": sum(c.isdigit() for c in code) / n_chars,
        # maintainability index (Halstead-lite variant)
        "maintainability": max(
            0.0,
            171
            - 5.2 * math.log(max(n_chars, 2))
            - 0.23 * (branches + 1)
            - 16.2 * math.log(max(n_lines, 2)),
        ),
    }


def extract(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([stylometry(c) for c in df["code"]], index=df.index).fillna(0.0)
