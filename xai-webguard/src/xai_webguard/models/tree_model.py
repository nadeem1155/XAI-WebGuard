"""Model A: gradient-boosted tree over engineered features.

Uses XGBoost when available (it exposes exact SHAP values through
``pred_contribs=True``, so no extra dependency is needed for TreeSHAP). Falls
back to scikit-learn's HistGradientBoosting when XGBoost is absent.
"""
from __future__ import annotations

import pickle
from typing import List, Optional

import numpy as np

from ..features import FEATURE_NAMES, extract_features, features_to_vector
from ..canonicalize import canonicalize

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier


class TreeModel:
    def __init__(self, params: Optional[dict] = None):
        self.params = params or {}
        self.booster = None
        self.sk = None
        self.feature_names = FEATURE_NAMES

    # ---- training ----
    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if _HAS_XGB:
            p = dict(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                     max_depth=self.params.get("max_depth", 8),
                     eta=self.params.get("learning_rate", 0.1),
                     subsample=0.9, colsample_bytree=0.9, seed=self.params.get("seed", 0))
            dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)
            evals = [(dtrain, "train")]
            if X_val is not None:
                dval = xgb.DMatrix(np.asarray(X_val, float), label=np.asarray(y_val, int),
                                   feature_names=self.feature_names)
                evals.append((dval, "val"))
            self.booster = xgb.train(p, dtrain,
                                     num_boost_round=self.params.get("n_estimators", 400),
                                     evals=evals,
                                     early_stopping_rounds=self.params.get("early_stopping", 30),
                                     verbose_eval=False)
        else:  # pragma: no cover
            self.sk = HistGradientBoostingClassifier(
                max_depth=self.params.get("max_depth", 8),
                learning_rate=self.params.get("learning_rate", 0.1),
                max_iter=self.params.get("n_estimators", 400))
            self.sk.fit(X, y)
        return self

    # ---- inference ----
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.booster is not None:
            d = xgb.DMatrix(X, feature_names=self.feature_names)
            return self.booster.predict(d)
        return self.sk.predict_proba(X)

    def predict_proba_raw(self, raw: str, field: str = "query") -> np.ndarray:
        vec = features_to_vector(extract_features(canonicalize(raw), field))
        return self.predict_proba(np.asarray([vec]))[0]

    # ---- exact SHAP contributions (TreeSHAP) ----
    def shap_values(self, X: np.ndarray, target_class: int) -> np.ndarray:
        """Return per-feature SHAP values for ``target_class``.

        With XGBoost this is exact TreeSHAP via ``pred_contribs``. The last
        column (bias) is dropped.
        """
        X = np.asarray(X, dtype=float)
        if self.booster is not None:
            d = xgb.DMatrix(X, feature_names=self.feature_names)
            contribs = self.booster.predict(d, pred_contribs=True)  # (n, class, feat+1)
            if contribs.ndim == 3:
                return contribs[:, target_class, :-1]
            return contribs[:, :-1]
        # fallback: permutation-style approximation
        base = self.predict_proba(X)[:, target_class]
        out = np.zeros((X.shape[0], X.shape[1]))
        for j in range(X.shape[1]):
            Xp = X.copy()
            Xp[:, j] = np.median(X[:, j])
            out[:, j] = base - self.predict_proba(Xp)[:, target_class]
        return out

    def save(self, path: str):
        with open(path, "wb") as fh:
            pickle.dump({"params": self.params,
                         "booster": self.booster.save_raw() if self.booster else None,
                         "sk": self.sk}, fh)

    @classmethod
    def load(cls, path: str) -> "TreeModel":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        m = cls(d["params"])
        if d["booster"] is not None and _HAS_XGB:
            m.booster = xgb.Booster()
            m.booster.load_model(bytearray(d["booster"]))
        m.sk = d.get("sk")
        return m
