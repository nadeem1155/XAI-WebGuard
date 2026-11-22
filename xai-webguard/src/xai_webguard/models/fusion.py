"""Calibrated fusion of the two detectors (Section 3.6, Eq. 3.1).

Each model's probabilities are temperature-scaled on the validation set, then
combined by a weighted average whose weight is also tuned on validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score


def _logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p)


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Find T>0 minimising NLL of temperature-scaled probabilities."""
    logits = _logit(probs)

    def nll(logT):
        T = np.exp(logT)
        p = _softmax(logits / T)
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))

    res = minimize_scalar(nll, bounds=(-2.0, 2.0), method="bounded")
    return float(np.exp(res.x))


@dataclass
class FusionModel:
    temp_a: float = 1.0
    temp_b: float = 1.0
    weight: float = 0.5           # weight on Model A
    uncertainty_band: float = 0.15  # fused top-prob within this of 2nd -> "flag for review"

    def calibrate(self, pa, ya, pb=None, yb=None):
        self.temp_a = fit_temperature(pa, ya)
        if pb is not None:
            self.temp_b = fit_temperature(pb, yb)
        return self

    def _scale(self, probs, T):
        return _softmax(_logit(probs) / T)

    def fuse(self, pa: np.ndarray, pb: Optional[np.ndarray]) -> np.ndarray:
        a = self._scale(pa, self.temp_a)
        if pb is None:
            return a
        b = self._scale(pb, self.temp_b)
        return self.weight * a + (1 - self.weight) * b

    def tune_weight(self, pa, pb, y, grid=None):
        """Pick the Model-A weight that maximises validation macro-F1."""
        grid = grid if grid is not None else np.linspace(0, 1, 21)
        best_w, best_f1 = 0.5, -1
        for w in grid:
            self.weight = float(w)
            pred = self.fuse(pa, pb).argmax(1)
            f1 = f1_score(y, pred, average="macro")
            if f1 > best_f1:
                best_f1, best_w = f1, float(w)
        self.weight = best_w
        return best_w

    def decide(self, fused: np.ndarray):
        """Return (label, confidence, review_flag) per row."""
        order = np.argsort(-fused, axis=1)
        top = order[:, 0]
        conf = fused[np.arange(len(fused)), top]
        second = fused[np.arange(len(fused)), order[:, 1]]
        review = (conf - second) < self.uncertainty_band
        return top, conf, review
