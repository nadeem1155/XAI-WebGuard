"""
Explainable AI (XAI) Module for XAI-WebGuard
Integrates SHAP (TreeExplainer) with the trained tree-based model.
Provides:
1. Global feature importance analysis (multi-class & per-attack type).
2. Local prediction explanations (waterfall/bar plots for individual requests).
3. Suspicious syntax and token highlighting with character offsets.
4. Natural language justification for security analysts.
"""

import os
import re
import html
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from .features import (
    SQL_KEYWORDS, SQL_OPERATORS, HTML_TAGS, EVENT_HANDLERS,
    SCRIPT_SCHEMES, DOM_SINKS, TAUTOLOGY_PATTERN
)

CLASS_NAMES = ["Benign Request", "SQL Injection", "Cross-Site Scripting"]
SHORT_CLASSES = ["Benign", "SQLi", "XSS"]


class WebGuardExplainer:
    """
    SHAP-based Explainer for the XAI-WebGuard detection system.
    """
    def __init__(self, model: Any, feature_pipeline: Any, feature_names: List[str]):
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.feature_names = feature_names
        
        # Initialize SHAP TreeExplainer
        print("[*] Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)

    def explain_global(
        self,
        X_sample: np.ndarray,
        output_dir: str = "results",
        top_k: int = 15
    ) -> Dict[str, Any]:
        """
        Computes global SHAP feature importance across a representative background set.
        Generates summary plots for:
        1. Overall global importance
        2. SQL Injection specific features
        3. XSS specific features
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"[*] Calculating SHAP values for global explanation on {len(X_sample)} samples...")
        shap_values = self.explainer(X_sample)
        
        # shap_values.values has shape (N, num_features, num_classes) or (N, num_features)
        raw_vals = shap_values.values
        if len(raw_vals.shape) == 2:
            # Binary or single output
            mean_abs_per_class = [np.mean(np.abs(raw_vals), axis=0)]
        else:
            mean_abs_per_class = [
                np.mean(np.abs(raw_vals[:, :, c]), axis=0) for c in range(raw_vals.shape[2])
            ]

        # 1. Plot Multi-Class Global Importance Bar Chart
        overall_importance = np.mean(np.abs(raw_vals), axis=(0, 2)) if len(raw_vals.shape) == 3 else np.mean(np.abs(raw_vals), axis=0)
        top_indices = np.argsort(overall_importance)[::-1][:top_k]

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=overall_importance[top_indices],
            y=[self.feature_names[i] for i in top_indices],
            palette="viridis"
        )
        plt.title(f"SHAP Global Feature Importance (Top {top_k} Features)", fontsize=13, fontweight="bold")
        plt.xlabel("Mean |SHAP Value| (Impact on Model Output Magnitude)", fontsize=10, fontweight="bold")
        plt.tight_layout()
        global_plot_path = os.path.join(output_dir, "shap_global_summary_bar.png")
        plt.savefig(global_plot_path, dpi=300)
        plt.close()
        print(f"[+] Saved global SHAP plot: {global_plot_path}")

        # 2. Plot SQL Injection Specific Features (Class 1)
        if len(mean_abs_per_class) > 1:
            sqli_imp = mean_abs_per_class[1]
            sqli_top_idx = np.argsort(sqli_imp)[::-1][:top_k]
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=sqli_imp[sqli_top_idx],
                y=[self.feature_names[i] for i in sqli_top_idx],
                palette="Reds_r"
            )
            plt.title(f"Top SHAP Features Driving SQL Injection Detection", fontsize=13, fontweight="bold")
            plt.xlabel("Mean |SHAP Value| for SQLi Class", fontsize=10, fontweight="bold")
            plt.tight_layout()
            sqli_plot_path = os.path.join(output_dir, "shap_sqli_importance.png")
            plt.savefig(sqli_plot_path, dpi=300)
            plt.close()
            print(f"[+] Saved SQLi SHAP plot: {sqli_plot_path}")

        # 3. Plot XSS Specific Features (Class 2)
        if len(mean_abs_per_class) > 2:
            xss_imp = mean_abs_per_class[2]
            xss_top_idx = np.argsort(xss_imp)[::-1][:top_k]
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=xss_imp[xss_top_idx],
                y=[self.feature_names[i] for i in xss_top_idx],
                palette="Blues_r"
            )
            plt.title(f"Top SHAP Features Driving Cross-Site Scripting (XSS) Detection", fontsize=13, fontweight="bold")
            plt.xlabel("Mean |SHAP Value| for XSS Class", fontsize=10, fontweight="bold")
            plt.tight_layout()
            xss_plot_path = os.path.join(output_dir, "shap_xss_importance.png")
            plt.savefig(xss_plot_path, dpi=300)
            plt.close()
            print(f"[+] Saved XSS SHAP plot: {xss_plot_path}")

        return {
            "top_global_features": [self.feature_names[i] for i in top_indices],
            "global_plot_path": global_plot_path,
            "sqli_plot_path": sqli_plot_path if len(mean_abs_per_class) > 1 else None,
            "xss_plot_path": xss_plot_path if len(mean_abs_per_class) > 2 else None
        }

    def explain_instance(
        self,
        raw_text: str,
        top_n: int = 8
    ) -> Dict[str, Any]:
        """
        Generates local explanation for a single raw input request.
        Returns:
        - Prediction, probabilities
        - Top contributing positive and negative features with SHAP scores
        - Suspicious highlighted terms
        - Natural language explanation text
        """
        # Transform input
        X_vec = self.feature_pipeline.transform([raw_text])
        if hasattr(X_vec, "toarray"):
            X_dense = X_vec.toarray()
        else:
            X_dense = np.array(X_vec)

        # Model prediction
        pred_class = int(self.model.predict(X_vec)[0])
        probabilities = self.model.predict_proba(X_vec)[0].tolist()
        pred_label = CLASS_NAMES[pred_class]
        confidence = probabilities[pred_class]

        # Compute SHAP values for single instance
        shap_res = self.explainer(X_dense)
        shap_vals = shap_res.values[0]  # shape (num_features, 3) or (num_features,)

        if len(shap_vals.shape) == 2:
            class_shap = shap_vals[:, pred_class]
        else:
            class_shap = shap_vals

        # Sort features by absolute contribution to the predicted class
        sorted_indices = np.argsort(np.abs(class_shap))[::-1]
        
        top_features = []
        for idx in sorted_indices[:top_n]:
            val = float(class_shap[idx])
            feature_name = self.feature_names[idx]
            clean_name = feature_name.replace("tfidf_", "Token: ").replace("domain_", "Feature: ")
            feature_val = float(X_dense[0, idx])
            top_features.append({
                "feature": clean_name,
                "raw_name": feature_name,
                "shap_value": val,
                "feature_value": feature_val,
                "impact": "Increases Attack Likelihood" if val > 0 else "Decreases Attack Likelihood"
            })

        # Identify suspicious terms in the raw text
        suspicious_terms = self.extract_suspicious_terms(raw_text)

        # Generate natural language justification
        explanation_narrative = self.generate_narrative(
            raw_text, pred_class, confidence, top_features, suspicious_terms
        )

        return {
            "prediction_class": pred_class,
            "prediction_label": pred_label,
            "confidence": confidence,
            "probabilities": {
                CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
            },
            "top_features": top_features,
            "suspicious_terms": suspicious_terms,
            "explanation_narrative": explanation_narrative
        }

    @staticmethod
    def extract_suspicious_terms(text: str) -> List[Dict[str, Any]]:
        """
        Detects specific suspicious words, tokens, and attack patterns in text
        along with their start and end character spans.
        """
        findings = []
        low = text.lower()

        # 1. Check SQL Keywords
        for kw in SQL_KEYWORDS:
            for m in re.finditer(r"\b" + re.escape(kw) + r"\b", low):
                findings.append({
                    "term": text[m.start():m.end()],
                    "category": "SQL Keyword",
                    "start": m.start(),
                    "end": m.end()
                })

        # 2. Check SQL Operators & Comment markers
        for op in SQL_OPERATORS:
            pos = 0
            while True:
                idx = low.find(op, pos)
                if idx == -1:
                    break
                findings.append({
                    "term": text[idx:idx + len(op)],
                    "category": "SQL Operator / Comment",
                    "start": idx,
                    "end": idx + len(op)
                })
                pos = idx + len(op)

        # 3. Check HTML Tags
        for tag in HTML_TAGS:
            pattern = r"<\s*/?\s*" + re.escape(tag) + r"\b"
            for m in re.finditer(pattern, low):
                findings.append({
                    "term": text[m.start():m.end()],
                    "category": "HTML Tag",
                    "start": m.start(),
                    "end": m.end()
                })

        # 4. Check JS Event Handlers
        for ev in EVENT_HANDLERS:
            for m in re.finditer(r"\b" + re.escape(ev) + r"\b", low):
                findings.append({
                    "term": text[m.start():m.end()],
                    "category": "JavaScript Event Handler",
                    "start": m.start(),
                    "end": m.end()
                })

        # 5. Check DOM Sinks
        for sink in DOM_SINKS:
            for m in re.finditer(re.escape(sink), low):
                findings.append({
                    "term": text[m.start():m.end()],
                    "category": "DOM Sink / Executable Function",
                    "start": m.start(),
                    "end": m.end()
                })

        # 6. Check Tautology
        for m in TAUTOLOGY_PATTERN.finditer(low):
            findings.append({
                "term": text[m.start():m.end()],
                "category": "SQL Tautology Pattern (e.g., 1=1)",
                "start": m.start(),
                "end": m.end()
            })

        # Deduplicate overlapping spans
        findings = sorted(findings, key=lambda x: (x["start"], -(x["end"] - x["start"])))
        filtered = []
        last_end = -1
        for item in findings:
            if item["start"] >= last_end:
                filtered.append(item)
                last_end = item["end"]

        return filtered

    @staticmethod
    def generate_narrative(
        text: str,
        pred_class: int,
        confidence: float,
        top_features: List[Dict[str, Any]],
        suspicious_terms: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesizes a clear, human-understandable explanation for why the model arrived at the decision.
        """
        pct = confidence * 100
        if pred_class == 0:
            return (
                f"The request is classified as **Benign** with **{pct:.1f}% confidence**. "
                f"The input lacks malicious control syntax, unauthorized script execution tags, or query alteration operators. "
                f"Feature attribution indicates typical query parameter patterns with balanced punctuation and low symbol entropy."
            )
        elif pred_class == 1:
            terms = [f"'{t['term']}' ({t['category']})" for t in suspicious_terms[:4]]
            terms_str = ", ".join(terms) if terms else "anomalous SQL delimiters"
            top_kw = [f['feature'] for f in top_features if f['shap_value'] > 0][:3]
            top_kw_str = ", ".join(top_kw) if top_kw else "SQL syntax markers"
            return (
                f"The request is classified as **SQL Injection (SQLi)** with **{pct:.1f}% confidence**. "
                f"The primary attack triggers identified include {terms_str}. "
                f"SHAP attribution confirms that the decision was driven by {top_kw_str}, which attempt "
                f"to manipulate relational database logic, bypass authentication checks, or extract structured data."
            )
        else:
            terms = [f"'{t['term']}' ({t['category']})" for t in suspicious_terms[:4]]
            terms_str = ", ".join(terms) if terms else "injected HTML/JavaScript tags"
            top_kw = [f['feature'] for f in top_features if f['shap_value'] > 0][:3]
            top_kw_str = ", ".join(top_kw) if top_kw else "client-side execution vectors"
            return (
                f"The request is classified as **Cross-Site Scripting (XSS)** with **{pct:.1f}% confidence**. "
                f"The detector intercepted active client-side script vectors including {terms_str}. "
                f"SHAP analysis indicates high positive attribution from {top_kw_str}, which allow an attacker "
                f"to execute malicious scripts within a victim's browser session to hijack credentials or cookies."
            )
