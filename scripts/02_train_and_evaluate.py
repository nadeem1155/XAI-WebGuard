"""
Script 02: End-to-End Training, Comparative Evaluation, and SHAP Explainability
Trains Logistic Regression, Random Forest, and XGBoost on the unified dataset,
evaluates all metrics, generates publication-quality figures, and computes SHAP explanations.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import HybridFeaturePipeline
from src.train_models import get_model_instances, train_and_benchmark_model, save_model_artifacts
from src.evaluate import (
    compute_metrics, plot_confusion_matrices, plot_roc_curves,
    plot_performance_comparison
)
from src.xai_explainer import WebGuardExplainer


def main():
    print("=" * 75)
    print("PHASES 2, 3, 4, 5: FEATURE EXTRACTION, TRAINING, EVALUATION, AND SHAP")
    print("=" * 75)

    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")
    models_dir = os.path.join("models")
    results_dir = os.path.join("results")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Data
    print("\n[Step 1] Loading preprocessed datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train_raw = train_df["payload"].astype(str).tolist()
    y_train = train_df["label"].values.astype(int)

    X_test_raw = test_df["payload"].astype(str).tolist()
    y_test = test_df["label"].values.astype(int)

    print(f"  Training samples: {len(X_train_raw):,}")
    print(f"  Testing samples : {len(X_test_raw):,}")

    # 2. Feature Extraction (TF-IDF + Domain Features)
    print("\n[Step 2] Fitting Hybrid Feature Pipeline (TF-IDF + Cybersecurity Domain Features)...")
    t0 = time.time()
    feature_pipeline = HybridFeaturePipeline(max_tfidf_features=2500)
    X_train = feature_pipeline.fit_transform(X_train_raw)
    X_test = feature_pipeline.transform(X_test_raw)
    feature_names = feature_pipeline.get_feature_names()
    print(f"  Feature Matrix Shape: {X_train.shape} extracted in {time.time() - t0:.2f}s")
    print(f"  Total features: {len(feature_names):,} (TF-IDF: {len(feature_names) - 15:,}, Domain: 15)")

    # 3. Model Training & Benchmarking
    print("\n[Step 3] Training and Benchmarking Classifiers...")
    model_instances = get_model_instances()
    benchmarks = {}
    evaluated_metrics = {}

    for model_name, model in model_instances.items():
        res = train_and_benchmark_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv_folds=5
        )
        benchmarks[model_name] = res

        # Compute detailed metrics
        metrics = compute_metrics(
            y_true=y_test,
            y_pred=res["y_pred"],
            y_proba=res["y_proba"],
            training_time=res["training_time_sec"],
            prediction_time=res["prediction_time_sec"],
            latency_ms=res["latency_per_sample_ms"]
        )
        metrics["cv_macro_f1"] = res["cv_mean_f1"]
        metrics["cv_std_f1"] = res["cv_std_f1"]
        evaluated_metrics[model_name] = metrics

    # 4. Generate Comparative Results Table
    print("\n[Step 4] Compiling Comparative Performance Table...")
    table_rows = []
    for name, m in evaluated_metrics.items():
        table_rows.append({
            "Model": name,
            "Accuracy": round(m["accuracy"], 4),
            "Macro Precision": round(m["precision_macro"], 4),
            "Macro Recall": round(m["recall_macro"], 4),
            "Macro F1": round(m["f1_macro"], 4),
            "Weighted F1": round(m["f1_weighted"], 4),
            "Macro ROC-AUC": round(m["roc_auc_macro"], 4),
            "5-Fold CV F1": f"{m['cv_macro_f1']:.4f} (+/- {m['cv_std_f1']:.4f})",
            "Training Time (s)": round(m["training_time_sec"], 2),
            "Inference Latency (ms/sample)": round(m["latency_per_sample_ms"], 3)
        })

    comparison_df = pd.DataFrame(table_rows)
    comparison_csv_path = os.path.join(results_dir, "comparison_table.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)

    print("\n" + "=" * 90)
    print("EXPERIMENTAL EVALUATION RESULTS (MEASURED DIRECTLY FROM EXECUTION)")
    print("=" * 90)
    print(comparison_df.to_string(index=False))
    print("=" * 90)

    # Save detailed JSON summary
    metrics_json_path = os.path.join(results_dir, "metrics_summary.json")
    with open(metrics_json_path, "w") as f:
        json.dump(evaluated_metrics, f, indent=2)
    print(f"\n[+] Detailed metrics saved to: {metrics_json_path}")

    # Identify Best Model based on Macro F1
    best_model_name = comparison_df.sort_values(by="Macro F1", ascending=False).iloc[0]["Model"]
    print(f"\n[🏆] Best Performing Model: {best_model_name} (Macro F1 = {evaluated_metrics[best_model_name]['f1_macro']:.4f})")

    # 5. Plotting Evaluation Figures
    print("\n[Step 5] Generating Publication-Quality Evaluation Charts...")
    plot_confusion_matrices(evaluated_metrics, os.path.join(results_dir, "confusion_matrices.png"))
    plot_roc_curves(benchmarks, y_test, os.path.join(results_dir, "roc_curves.png"))
    plot_performance_comparison(comparison_df, os.path.join(results_dir, "performance_comparison.png"))

    # 6. Save Trained Artifacts
    print("\n[Step 6] Saving Models and Pipelines to Disk...")
    artifacts = {
        "hybrid_feature_pipeline": feature_pipeline,
        "logistic_regression": benchmarks["Logistic Regression"]["model"],
        "random_forest": benchmarks["Random Forest"]["model"],
        "xgboost": benchmarks["XGBoost"]["model"]
    }
    save_model_artifacts(artifacts, models_dir)

    # 7. SHAP Explainability Integration
    print("\n[Step 7] Computing SHAP Global Explainability with Best Tree Model...")
    # Use best tree model (XGBoost or Random Forest)
    best_tree_model = benchmarks["XGBoost"]["model"]
    explainer = WebGuardExplainer(
        model=best_tree_model,
        feature_pipeline=feature_pipeline,
        feature_names=feature_names
    )

    # Sample representative background subset for global SHAP to maintain high speed
    np.random.seed(42)
    sample_indices = np.random.choice(X_test.shape[0], size=min(X_test.shape[0], 600), replace=False)
    X_sample_dense = X_test[sample_indices].toarray()

    shap_summary = explainer.explain_global(X_sample_dense, output_dir=results_dir, top_k=15)
    print("  Top 10 Global SHAP Features:", shap_summary["top_global_features"][:10])

    # 8. Test Sample Inferences with Local Explanations
    print("\n[Step 8] Testing Live Local Inferences with Explanations...")
    test_queries = [
        ("id=105 OR 1=1--", "Sample SQL Injection"),
        ("<script>alert('Session Hijack: ' + document.cookie)</script>", "Sample Stored XSS"),
        ("/products/catalog?category=electronics&sort=price_asc&page=2", "Sample Benign Web Request"),
        ("<img src=x onerror=alert(1)>", "Sample Image XSS"),
        ("admin' UNION SELECT username, password FROM users--", "Sample Union SQLi")
    ]

    for query, description in test_queries:
        exp = explainer.explain_instance(query, top_n=5)
        print(f"\n[Test Case] {description}")
        print(f"  Input       : {query}")
        print(f"  Prediction  : {exp['prediction_label']} ({exp['confidence']*100:.1f}%)")
        print(f"  Suspicious  : {[t['term'] for t in exp['suspicious_terms']]}")
        print(f"  Explanation : {exp['explanation_narrative']}")
        top_str = [f"{f['feature']} ({f['shap_value']:+.3f})" for f in exp['top_features'][:3]]
        print(f"  Top SHAP    : {top_str}")

    print("\n" + "=" * 75)
    print("ALL PHASES (DATASET, TRAINING, EVALUATION, SHAP) EXECUTED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
