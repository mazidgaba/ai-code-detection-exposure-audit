"""FastAPI service.

Every machine/adversarial verdict carries a caveat string. That field is
required, not optional: AICD Bench measured every published detector at or
below random under genuine distribution shift, so a bare verdict from this
system would misrepresent what it knows.
"""
from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aicd import config as C
from aicd.serve.runtime import load_runtime

app = FastAPI(
    title="AI Code Provenance",
    version="0.1.0",
    description="Provenance evidence for source code. Not a verdict machine.",
)

_rt = None
_sema = asyncio.Semaphore(8)

LANGS = ["python", "java", "cpp", "c", "csharp", "go", "javascript", "php"]


def rt():
    global _rt
    if _rt is None:
        _rt = load_runtime(C.load("base.yaml"))
        if not _rt.available():
            raise RuntimeError("no trained branch on disk; run models/xgb.py")
    return _rt


class AnalyzeIn(BaseModel):
    code: str = Field(..., min_length=1, max_length=400_000)
    language: str = "python"
    attribute_lines: bool = Field(
        False, description="also return the line ranges that drove the call")


class BatchIn(BaseModel):
    items: list[AnalyzeIn] = Field(..., max_length=256)


class AnalyzeOut(BaseModel):
    decision: Literal["human", "machine", "hybrid", "adversarial", "abstain"]
    p_machine: float
    probabilities: dict
    confidence_note: str
    branches_used: list[str]
    thresholds: dict
    flagged_regions: list[dict] = []


@app.get("/health")
def health():
    r = rt()
    return {
        "status": "ok",
        "branches": r.available(),
        "thresholds": r.thresholds,
        "calibrated_languages": sorted(r.isotonic.keys()),
        "classes": ["human", "machine", "hybrid", "adversarial"],
    }


def _analyze(item: AnalyzeIn) -> dict:
    r = rt()
    lang = item.language if item.language in LANGS else "python"
    out = r.analyze(item.code, lang)
    out["flagged_regions"] = []

    if item.attribute_lines and out["decision"] in ("hybrid", "machine", "adversarial"):
        from aicd.serve.attribution import per_line_scores, score_windows, top_regions

        wins, pm = score_windows(item.code, lambda cs: r.score_batch(cs, [lang] * len(cs)))
        if len(wins):
            ls = per_line_scores(item.code, wins, pm)
            out["flagged_regions"] = [
                {"start_line": s, "end_line": e, "p_machine": round(p, 4)}
                for s, e, p in top_regions(ls, r.thresholds["t_high"])
            ]
    return out


@app.post("/analyze", response_model=AnalyzeOut)
async def analyze(item: AnalyzeIn):
    try:
        async with _sema:
            return await asyncio.to_thread(_analyze, item)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/analyze/batch")
async def analyze_batch(body: BatchIn):
    async with _sema:
        results = await asyncio.to_thread(lambda: [_analyze(i) for i in body.items])
    return {"count": len(results), "results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
