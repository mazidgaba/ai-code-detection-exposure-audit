"""tree-sitter AST features: depth, node counts, per-node-type densities.

CoDet-M4's SHAP analysis put AST depth among the top discriminative signals,
so this branch is not optional decoration.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

TS_LANG = {
    "python": "python", "java": "java", "cpp": "cpp", "c": "c",
    "csharp": "c_sharp", "go": "go", "javascript": "javascript",
    "php": "php", "ruby": "ruby", "rust": "rust",
}


@lru_cache(maxsize=32)
def get_parser(language: str) -> Any | None:
    """Return a tree-sitter parser, or None if the language is unavailable."""
    name = TS_LANG.get(language)
    if name is None:
        return None
    try:
        from tree_sitter_language_pack import get_parser as _gp
        return _gp(name)
    except Exception:
        try:
            from tree_sitter_languages import get_parser as _gp2
            return _gp2(name)
        except Exception:
            return None


def parse_ok(parser: Any | None, code: str) -> bool:
    """True if the sample parses without error. Missing parser = pass through."""
    if parser is None:
        return True
    try:
        tree = parser.parse(code.encode("utf-8", "ignore"))
    except Exception:
        return False
    return not tree.root_node.has_error


def _walk(node, depth: int, counter: Counter, depths: list[int]):
    counter[node.type] += 1
    depths.append(depth)
    for ch in node.children:
        _walk(ch, depth + 1, counter, depths)


def ast_features(code: str, language: str) -> dict[str, float]:
    parser = get_parser(language)
    base = {"ast_max_depth": 0.0, "ast_mean_depth": 0.0, "ast_nodes": 0.0, "ast_parse_ok": 0.0}
    if parser is None:
        return base
    try:
        tree = parser.parse(code.encode("utf-8", "ignore"))
    except Exception:
        return base

    counter: Counter = Counter()
    depths: list[int] = []
    try:
        _walk(tree.root_node, 0, counter, depths)
    except RecursionError:
        return base

    n = max(len(depths), 1)
    feats = {
        "ast_max_depth": float(max(depths) if depths else 0),
        "ast_mean_depth": float(np.mean(depths) if depths else 0),
        "ast_nodes": float(n),
        "ast_parse_ok": 0.0 if tree.root_node.has_error else 1.0,
    }
    # Node-type densities, normalized by total node count so long files
    # don't dominate. Language-specific type names are fine here; the
    # >20% missing filter downstream drops types that rarely appear.
    for t, c in counter.items():
        feats[f"ast_d_{t}"] = c / n
    return feats


def extract(df: pd.DataFrame) -> pd.DataFrame:
    rows = [ast_features(c, l) for c, l in zip(df["code"], df["language"])]
    return pd.DataFrame(rows, index=df.index).fillna(0.0)
