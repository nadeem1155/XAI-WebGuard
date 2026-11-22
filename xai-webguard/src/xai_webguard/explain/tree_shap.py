"""Feature attributions for Model A (TreeSHAP), with ranked, named output."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..features import FEATURE_NAMES, extract_features, features_to_vector
from ..canonicalize import canonicalize


def feature_attributions(tree_model, raw: str, target_class: int, field: str = "query",
                         top_k: int = 8) -> List[Dict]:
    """Return the top-k named feature attributions for one input."""
    vec = np.asarray([features_to_vector(extract_features(canonicalize(raw), field))])
    sv = tree_model.shap_values(vec, target_class)[0]
    order = np.argsort(-np.abs(sv))[:top_k]
    return [{"feature": FEATURE_NAMES[i], "value": float(vec[0][i]),
             "shap": float(sv[i])} for i in order]


def global_importance(tree_model, X: np.ndarray, target_class: int) -> List[Dict]:
    """Mean |SHAP| per feature across a sample X (artefact-spotting)."""
    sv = tree_model.shap_values(np.asarray(X, float), target_class)
    mean_abs = np.mean(np.abs(sv), axis=0)
    order = np.argsort(-mean_abs)
    return [{"feature": FEATURE_NAMES[i], "mean_abs_shap": float(mean_abs[i])} for i in order]
