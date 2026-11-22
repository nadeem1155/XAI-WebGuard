"""End-to-end XAI-WebGuard pipeline: canonicalise -> detect -> explain -> report."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from .canonicalize import canonicalize
from .features import extract_features, features_to_vector
from .models.fusion import FusionModel
from .models.tree_model import TreeModel
from .explain.tree_shap import feature_attributions
from .explain.semantic_mapper import map_spans, dominant_standard

CLASS_NAMES = ["benign", "sqli", "xss"]


@dataclass
class Report:
    verdict: str
    confidence: float
    action: str
    review: bool
    evidence: List[Dict]
    standards: Dict
    model_agreement: Optional[bool]
    raw: str

    def to_json(self, **kw):
        return json.dumps(asdict(self), **kw)


class XAIWebGuard:
    def __init__(self, tree: TreeModel, transformer=None, fusion: Optional[FusionModel] = None,
                 block_threshold: float = 0.9):
        self.tree = tree
        self.transformer = transformer
        self.fusion = fusion or FusionModel()
        self.block_threshold = block_threshold

    def _proba(self, raw: str, field: str):
        vec = np.asarray([features_to_vector(extract_features(canonicalize(raw), field))])
        pa = self.tree.predict_proba(vec)
        pb = None
        if self.transformer is not None:
            try:
                pb = self.transformer.predict_proba([raw])
            except Exception:
                pb = None
        return pa, pb

    def inspect(self, raw: str, field: str = "query", explain: bool = True) -> Report:
        pa, pb = self._proba(raw, field)
        fused = self.fusion.fuse(pa, pb)
        label, conf, review = self.fusion.decide(fused)
        cls = int(label[0])
        name = CLASS_NAMES[cls]
        conf = float(conf[0])
        review = bool(review[0])
        agreement = None
        if pb is not None:
            agreement = bool(pa.argmax(1)[0] == pb.argmax(1)[0])

        action = "allow"
        if name != "benign":
            action = "block" if (conf >= self.block_threshold and not review) else "flag"

        evidence, standards = [], {"cwe": None, "capec": None}
        if explain and name != "benign":
            feats = feature_attributions(self.tree, raw, cls, field=field, top_k=6)
            spans = []
            for fa in feats:
                spans.append({"token": fa["feature"], "score": fa["shap"], "raw_text": _feature_text(fa["feature"], raw)})
            mapped = map_spans(spans)
            evidence = [{"feature": fa["feature"], "value": fa["value"], "attribution": fa["shap"],
                         "role": m.get("role"), "cwe": m.get("cwe"), "capec": m.get("capec")}
                        for fa, m in zip(feats, mapped)]
            standards = dominant_standard(mapped, name)
        return Report(verdict=name, confidence=conf, action=action, review=review,
                      evidence=evidence, standards=standards, model_agreement=agreement, raw=raw)


def _feature_text(feature: str, raw: str) -> str:
    """Best-effort snippet of the raw request that a feature points at."""
    import re
    patterns = {
        "tautology_pattern": r"(\bor\b|\band\b)\s*['\"]?\s*\d+\s*=\s*\d+",
        "comment_terminator": r"(--|#|/\*)",
        "union_present": r"\bunion\b",
        "event_handler_count": r"on[a-z]+\s*=",
        "script_scheme_count": r"(javascript:|data:|vbscript:)",
        "dom_sink_count": r"(alert|prompt|eval|innerhtml)",
        "html_tag_count": r"<\s*[a-z]+",
    }
    rx = patterns.get(feature)
    if rx:
        m = re.search(rx, raw, re.I)
        if m:
            return m.group(0)
    return feature
