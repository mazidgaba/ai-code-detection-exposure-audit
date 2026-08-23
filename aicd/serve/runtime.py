"""Runtime: load whichever branches are on disk and score code with them.

Deliberately tolerant -- the system is useful with branch B alone, and branch A
needs a GPU to train. Whatever is present gets used; the response says which.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aicd import config as C


@dataclass
class Runtime:
    cfg: object
    branch_b: dict | None = None
    branch_a: object | None = None
    tokenizer: object | None = None
    device: object = None
    stacker: object = None
    isotonic: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=lambda: {"t_high": 0.5, "t_low": 0.35})

    # ---------------------------------------------------------------- branches
    def available(self) -> list[str]:
        out = []
        if self.branch_b:
            out.append("b_xgb")
        if self.branch_a is not None:
            out.append("a_modernbert")
        if self.stacker is not None:
            out.append("stacker")
        return out

    def _proba_b(self, codes: list[str], languages: list[str]) -> np.ndarray | None:
        if not self.branch_b:
            return None
        from aicd.features.build import dense_features

        df = pd.DataFrame({"code": codes, "language": languages})
        dense = dense_features(df)
        from aicd.models.xgb import predict as xgb_predict

        return xgb_predict(self.branch_b["vec"], self.branch_b["clf"],
                           self.branch_b["dense_cols"], codes, dense)

    def _proba_a(self, codes: list[str]) -> np.ndarray | None:
        if self.branch_a is None:
            return None
        from aicd.models.modernbert_triplet import predict

        return predict(self.branch_a, self.tokenizer, codes, self.cfg, self.device)

    # ------------------------------------------------------------------ public
    def score_batch(self, codes: list[str], languages: list[str] | None = None) -> np.ndarray:
        languages = languages or ["python"] * len(codes)
        pa, pb = self._proba_a(codes), self._proba_b(codes, languages)

        if self.stacker is not None and pa is not None and pb is not None:
            X = np.hstack([pa, pb, np.zeros((len(codes), 1), np.float32)])
            return self.stacker.predict_proba(X)
        for p in (pa, pb):
            if p is not None:
                return p
        raise RuntimeError("no trained branch found; run models/xgb.py first")

    def calibrate(self, p_machine: np.ndarray, languages: list[str]) -> np.ndarray:
        if not self.isotonic:
            return p_machine
        from aicd.eval.calibration import apply_isotonic

        return apply_isotonic(self.isotonic, languages, p_machine)

    def analyze(self, code: str, language: str = "python") -> dict:
        from aicd.serve.policy import decide

        proba = self.score_batch([code], [language])[0]
        pm = self.calibrate(np.array([1.0 - proba[0]]), [language])[0]
        adjusted = proba.copy()
        adjusted[0] = 1.0 - pm
        if proba[1:].sum() > 0:
            adjusted[1:] = proba[1:] / proba[1:].sum() * pm

        d = decide(adjusted, self.thresholds)
        return {
            "decision": d.decision,
            "p_machine": round(float(pm), 4),
            "probabilities": {k: round(v, 4) for k, v in d.probabilities.items()},
            "confidence_note": d.confidence_note,
            "branches_used": self.available(),
            "thresholds": self.thresholds,
        }


def load_runtime(cfg=None) -> Runtime:
    cfg = cfg or C.load("base.yaml")
    art = C.artifacts(cfg)
    rt = Runtime(cfg=cfg)

    p = art / "xgb_branch_b.pkl"
    if p.exists():
        with open(p, "rb") as f:
            rt.branch_b = pickle.load(f)

    p = art / "branch_a_base.pt"
    if p.exists():
        try:
            import torch

            from aicd.models.modernbert_triplet import build, device_of

            ck = torch.load(p, map_location="cpu", weights_only=False)
            tok, model = build(cfg)
            model.load_state_dict(ck["state_dict"])
            dev = device_of()
            model.to(dev).eval()
            rt.branch_a, rt.tokenizer, rt.device = model, tok, dev
        except Exception as e:  # a missing GPU or torch build must not break serving
            print(f"[runtime] branch A unavailable: {type(e).__name__}: {e}")

    p = art / "stacker.pkl"
    if p.exists() and rt.branch_a is not None:
        with open(p, "rb") as f:
            rt.stacker = pickle.load(f)

    for b in ("stack", "b", "a"):
        p = art / f"isotonic_{b}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                rt.isotonic = pickle.load(f)
            break

    for b in ("stack", "b", "a"):
        p = art / f"policy_{b}.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                rt.thresholds = json.load(f)
            break

    return rt
