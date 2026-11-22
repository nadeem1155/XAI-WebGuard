"""Explanation-quality metrics (Section 4.6 / Chapter 5 of the thesis):
comprehensiveness, sufficiency, stability (Jaccard), plausibility (token-F1),
inter-explainer agreement (Kendall tau) and a model-randomisation sanity check.
"""
from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np


def comprehensiveness(prob_full: float, prob_without_evidence: float) -> float:
    """Drop in class probability when the top-k evidence is removed."""
    return float(prob_full - prob_without_evidence)


def sufficiency(prob_full: float, prob_only_evidence: float) -> float:
    """Drop in class probability when only the evidence is kept."""
    return float(prob_full - prob_only_evidence)


def topk_jaccard(tokens_a: Sequence, tokens_b: Sequence) -> float:
    a, b = set(tokens_a), set(tokens_b)
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def token_f1(pred_tokens: Sequence[int], gold_tokens: Sequence[int]) -> float:
    """F1 overlap between predicted top-k token indices and annotated span."""
    pred, gold = set(pred_tokens), set(gold_tokens)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    prec = tp / len(pred)
    rec = tp / len(gold)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def kendall_tau(rank_a: Sequence[float], rank_b: Sequence[float]) -> float:
    from scipy.stats import kendalltau
    if len(rank_a) < 2:
        return float("nan")
    t, _ = kendalltau(rank_a, rank_b)
    return float(t) if t == t else 0.0


def erasure_scores(model_prob_fn: Callable[[str], float], text: str,
                   token_spans: List[tuple], top_idx: Sequence[int]) -> dict:
    """Compute comprehensiveness & sufficiency given a scoring function.

    model_prob_fn(masked_text) -> probability of the target class.
    token_spans: char spans of each token in ``text``.
    top_idx: indices (into token_spans) of the top-k evidence tokens.
    """
    full = model_prob_fn(text)
    keep = set(top_idx)
    # remove evidence
    removed = _mask(text, [token_spans[i] for i in range(len(token_spans)) if i in keep])
    only = _mask(text, [token_spans[i] for i in range(len(token_spans)) if i not in keep])
    return {"comprehensiveness": comprehensiveness(full, model_prob_fn(removed)),
            "sufficiency": sufficiency(full, model_prob_fn(only))}


def _mask(text: str, spans: List[tuple], fill: str = " ") -> str:
    chars = list(text)
    for s, e in spans:
        for i in range(max(0, s), min(len(chars), e)):
            chars[i] = fill
    return "".join(chars)


def sanity_passes(attr_normal: np.ndarray, attr_random: np.ndarray, thresh: float = 0.3) -> bool:
    """A method passes if attributions change substantially under model
    randomisation (low correlation between the two attribution vectors)."""
    a, b = np.asarray(attr_normal).ravel(), np.asarray(attr_random).ravel()
    if a.std() == 0 or b.std() == 0:
        return True
    corr = abs(np.corrcoef(a, b)[0, 1])
    return corr < thresh
