"""
Model Evaluation Module for XAI-WebGuard
Computes multi-class metrics: Accuracy, Precision, Recall, F1-Score,
Confusion Matrix, ROC-AUC (One-vs-Rest), Training & Prediction time.
Generates comprehensive visualization figures.
"""

import os
import json
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ["Benign Request", "SQL Injection", "Cross-Site Scripting"]
SHORT_CLASSES = ["Benign", "SQLi", "XSS"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    training_time: float,
    prediction_time: float,
    latency_ms: float
) -> Dict[str, Any]:
    """Computes all standard machine learning performance metrics for multi-class classification."""
    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class precision, recall, F1
    p_per_class = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    r_per_class = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    # Multi-class ROC-AUC (One-vs-Rest)
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fpr = dict()
    tpr = dict()
    roc_auc_per_class = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc_per_class[SHORT_CLASSES[i]] = float(auc(fpr[i], tpr[i]))
    roc_auc_macro = float(np.mean(list(roc_auc_per_class.values())))

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "precision_weighted": p_weighted,
        "recall_macro": r_macro,
        "recall_weighted": r_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_per_class": roc_auc_per_class,
        "per_class": {
            SHORT_CLASSES[i]: {
                "precision": float(p_per_class[i]),
                "recall": float(r_per_class[i]),
                "f1": float(f1_per_class[i]),
                "roc_auc": float(roc_auc_per_class[SHORT_CLASSES[i]])
            } for i in range(3)
        },
        "confusion_matrix": cm,
        "training_time_sec": float(training_time),
        "prediction_time_sec": float(prediction_time),
        "latency_per_sample_ms": float(latency_ms)
    }


def plot_confusion_matrices(
    results: Dict[str, Dict[str, Any]],
    output_path: str = "results/confusion_matrices.png"
) -> None:
    """Plots confusion matrices for all three models side by side."""
    sns.set_theme(style="white", font="sans-serif")
    model_names = list(results.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))

    if len(model_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        cm = np.array(results[name]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=SHORT_CLASSES,
            yticklabels=SHORT_CLASSES,
            ax=ax,
            annot_kws={"size": 12, "weight": "bold"}
        )
        ax.set_title(f"{name}\nAccuracy: {results[name]['accuracy']*100:.2f}%", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=10, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrices: {output_path}")


def plot_roc_curves(
    results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    output_path: str = "results/roc_curves.png"
) -> None:
    """Plots multiclass ROC curves for all evaluated models."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    for idx, class_name in enumerate(SHORT_CLASSES):
        ax = axes[idx]
        for model_name, res in results.items():
            y_proba = res["y_proba"]
            fpr, tpr, _ = roc_curve(y_bin[:, idx], y_proba[:, idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=10, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=10, fontweight="bold")
        ax.set_title(f"ROC Curve: {class_name}", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved ROC curves: {output_path}")


def plot_performance_comparison(
    comparison_df: pd.DataFrame,
    output_path: str = "results/performance_comparison.png"
) -> None:
    """Plots comparative bar charts across all evaluation metrics."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    metrics_to_plot = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Macro ROC-AUC"]

    df_melted = comparison_df.melt(
        id_vars=["Model"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score"
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = ["#3498db", "#2ecc71", "#e67e22"]
    bar = sns.barplot(
        data=df_melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette=palette,
        ax=ax
    )
    ax.set_title("Machine Learning Models Comparative Performance", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax.set_xlabel("Evaluation Metric", fontsize=11, fontweight="bold")
    ax.legend(title="Algorithm", fontsize=10, title_fontsize=11)

    for p in bar.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", xytext=(0, 4),
                        textcoords="offset points", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved performance comparison: {output_path}")
