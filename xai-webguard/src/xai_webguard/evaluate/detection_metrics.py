"""Detection metrics: per-class P/R/F1, macro-F1, MCC, PR-AUC, ROC-AUC,
FPR at a target recall, and expected calibration error (Section 4.6)."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score)

LABELS = [0, 1, 2]
NAMES = ["benign", "sqli", "xss"]


def detection_report(y_true: np.ndarray, proba: np.ndarray,
                     target_recall: float = 0.99) -> Dict:
    y_true = np.asarray(y_true)
    y_pred = proba.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    rep = {"per_class": {NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])} for i in LABELS},
           "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
           "accuracy": float((y_pred == y_true).mean()),
           "mcc": float(matthews_corrcoef(y_true, y_pred)),
           "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()}
    # one-vs-rest AUCs (attack = any non-benign)
    try:
        attack_true = (y_true != 0).astype(int)
        attack_score = 1.0 - proba[:, 0]
        rep["pr_auc"] = float(average_precision_score(attack_true, attack_score))
        rep["roc_auc"] = float(roc_auc_score(attack_true, attack_score))
        rep["fpr_at_recall"] = float(fpr_at_recall(attack_true, attack_score, target_recall))
    except ValueError:
        rep["pr_auc"] = rep["roc_auc"] = rep["fpr_at_recall"] = None
    rep["ece"] = float(expected_calibration_error(y_true, proba))
    return rep


def fpr_at_recall(y_true_bin: np.ndarray, score: np.ndarray, target_recall: float) -> float:
    pos = score[y_true_bin == 1]
    neg = score[y_true_bin == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(pos, 1 - target_recall)   # threshold giving ~target recall
    return float((neg >= thr).mean())


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    conf = proba.max(1)
    pred = proba.argmax(1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece, n = 0.0, len(y_true)
    for b in range(bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return ece
