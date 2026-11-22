"""Integrated Gradients for the transformer (Eq. 3.3).

Implemented directly with torch autograd so Captum is not required. Attributions
are computed over input embeddings and summed to per-token scores, which are
then projected back to the raw request through the canonical offset map.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..canonicalize import canonicalize


def token_attributions(model, text: str, target_class: int, steps: int = 32):
    """Return (tokens, scores, canon) for one input.

    ``model`` is a TransformerModel with a loaded HF model and tokenizer.
    """
    import torch
    model._ensure_loaded()
    canon = canonicalize(text)
    enc = model.tokenizer(canon.text, truncation=True, max_length=model.max_len,
                          return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    mask = enc.get("attention_mask")
    embed_layer = model.model.get_input_embeddings()
    inp_emb = embed_layer(input_ids).detach()
    baseline = torch.zeros_like(inp_emb)

    total = torch.zeros_like(inp_emb)
    for alpha in np.linspace(1.0 / steps, 1.0, steps):
        point = (baseline + float(alpha) * (inp_emb - baseline)).clone().requires_grad_(True)
        out = model.model(inputs_embeds=point, attention_mask=mask)
        score = out.logits[0, target_class]
        grad = torch.autograd.grad(score, point)[0]
        total += grad.detach()
    avg_grad = total / steps
    ig = ((inp_emb - baseline) * avg_grad).sum(-1)[0]      # per sub-word token
    scores = ig.cpu().numpy()
    tokens = model.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    # normalise to [-1, 1]
    denom = np.max(np.abs(scores)) or 1.0
    scores = scores / denom
    return tokens, scores, canon


def top_raw_spans(model, text: str, target_class: int, k: int = 5, steps: int = 32):
    """Top-k attributed sub-word tokens mapped to raw-request character spans."""
    tokens, scores, canon = token_attributions(model, text, target_class, steps)
    order = np.argsort(-scores)
    # rebuild char offsets of each sub-word in the canonical string
    enc = model.tokenizer(canon.text, truncation=True, max_length=model.max_len,
                          return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    spans = []
    for idx in order:
        if idx >= len(offs):
            continue
        cs, ce = offs[idx]
        if ce <= cs:
            continue
        rs, re_ = canon.raw_span(cs, ce)
        spans.append({"token": tokens[idx], "score": float(scores[idx]),
                      "canon_span": (cs, ce), "raw_span": (rs, re_),
                      "raw_text": canon.raw[rs:re_]})
        if len(spans) >= k:
            break
    return spans
