"""
Model Training Module for XAI-WebGuard
Trains and evaluates:
1. Logistic Regression
2. Random Forest
3. XGBoost
Performs 5-Fold Stratified Cross-Validation, records training time, and saves trained models.
"""

import os
import time
import joblib
from typing import Dict, Tuple, Any
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def get_model_instances() -> Dict[str, Any]:
    """Returns initialized instances of the three target classifiers."""
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )
    }
    return models


def train_and_benchmark_model(
    model_name: str,
    model: Any,
    X_train: sp.csr_matrix,
    y_train: np.ndarray,
    X_test: sp.csr_matrix,
    y_test: np.ndarray,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Trains a model, performs cross-validation, and measures training time & inference latency.
    """
    print(f"\n--- Training {model_name} ---")

    # 1. 5-Fold Stratified Cross-Validation on Training set
    print(f"  Performing {cv_folds}-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    print(f"  CV Macro-F1: {cv_mean:.4f} (+/- {cv_std:.4f})")

    # 2. Fit on entire training set and measure time
    t_start = time.perf_counter()
    model.fit(X_train, y_train)
    t_train = time.perf_counter() - t_start
    print(f"  Full training time: {t_train:.2f} seconds")

    # 3. Benchmark prediction latency on test set
    t_pred_start = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    t_pred_total = time.perf_counter() - t_pred_start
    latency_per_sample_ms = (t_pred_total / len(y_test)) * 1000.0
    print(f"  Test prediction time: {t_pred_total:.3f} s ({latency_per_sample_ms:.3f} ms/sample)")

    return {
        "model_name": model_name,
        "model": model,
        "cv_mean_f1": cv_mean,
        "cv_std_f1": cv_std,
        "cv_scores": cv_scores.tolist(),
        "training_time_sec": t_train,
        "prediction_time_sec": t_pred_total,
        "latency_per_sample_ms": latency_per_sample_ms,
        "y_pred": y_pred,
        "y_proba": y_proba
    }


def save_model_artifacts(
    artifacts: Dict[str, Any],
    output_dir: str = "models"
) -> None:
    """Saves trained model objects to disk."""
    os.makedirs(output_dir, exist_ok=True)
    for name, obj in artifacts.items():
        save_path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(obj, save_path)
        print(f"[+] Saved {name} -> {save_path}")
