"""FastAPI prototype exposing XAI-WebGuard as reverse-proxy middleware.

Run (after training a model):
    uvicorn xai_webguard.serve.app:app --reload

The service never returns counterfactuals to the client; the /inspect endpoint
returns only the verdict, confidence and (optionally) the evidence report.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from ..models.tree_model import TreeModel
from ..models.fusion import FusionModel
from ..pipeline import XAIWebGuard

app = FastAPI(title="XAI-WebGuard", version="0.1.0")
_guard: Optional[XAIWebGuard] = None


class InspectRequest(BaseModel):
    input: str
    field: str = "query"
    explain: bool = True


def get_guard() -> XAIWebGuard:
    global _guard
    if _guard is None:
        tree_path = os.environ.get("XWG_TREE", "results/model_a.pkl")
        tree = TreeModel.load(tree_path)
        transformer = None
        tdir = os.environ.get("XWG_TRANSFORMER")
        if tdir and os.path.isdir(tdir):
            from ..models.transformer_model import TransformerModel
            transformer = TransformerModel.load(tdir)
        fusion = FusionModel()
        _guard = XAIWebGuard(tree, transformer, fusion)
    return _guard


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inspect")
def inspect(req: InspectRequest):
    guard = get_guard()
    report = guard.inspect(req.input, field=req.field, explain=req.explain)
    return report.__dict__
