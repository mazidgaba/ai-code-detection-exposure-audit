"""Feature extraction sanity."""
from __future__ import annotations

import pandas as pd

from aicd.features import ast_feats, stylometry

HUMAN = "def f(x):\n  if x>0: return x\n  return -x\n"
SPACED = "def compute_absolute_value(value):\n\n    if value > 0:\n\n        return value\n\n    return -value\n"


def test_stylometry_keys_stable():
    a = stylometry.stylometry(HUMAN)
    b = stylometry.stylometry(SPACED)
    assert set(a) == set(b)
    assert all(isinstance(v, (int, float)) for v in a.values())


def test_empty_line_ratio_discriminates():
    # The CoDet-M4 SHAP finding: generated code is spaced out more.
    assert stylometry.stylometry(SPACED)["r_empty_lines"] > stylometry.stylometry(HUMAN)["r_empty_lines"]


def test_ast_depth_positive_for_python():
    f = ast_feats.ast_features(SPACED, "python")
    assert f["ast_nodes"] >= 1


def test_extract_frames_align():
    df = pd.DataFrame({"code": [HUMAN, SPACED], "language": ["python", "python"]})
    s, a = stylometry.extract(df), ast_feats.extract(df)
    assert len(s) == len(a) == 2
    assert not s.isna().any().any()
