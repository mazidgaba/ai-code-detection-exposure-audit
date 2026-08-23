"""Config loading. YAML -> nested dict with attribute access, plus `extends` support."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent


class Cfg(dict):
    """dict with attribute access, recursive."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load(name: str = "base.yaml") -> Cfg:
    path = ROOT / "configs" / name if not os.path.isabs(name) else Path(name)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("extends", None)
    if parent:
        raw = _deep_merge(load(parent), raw)
    cfg = Cfg(raw)
    cfg["_root"] = str(ROOT)
    return cfg


def artifacts(cfg: Cfg) -> Path:
    p = ROOT / cfg.project.artifacts_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports(cfg: Cfg) -> Path:
    p = ROOT / cfg.project.reports_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


LABEL_NAMES = ["human", "machine", "hybrid", "adversarial"]
