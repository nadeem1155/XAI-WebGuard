"""
XAI-WebGuard: An Explainable Machine Learning-Based Framework for
Detecting SQL Injection and Cross-Site Scripting Attacks.
"""

from .preprocessing import clean_and_normalize
from .features import build_tfidf_vectorizer, DomainFeatureExtractor, HybridFeaturePipeline
from .train_models import get_model_instances, train_and_benchmark_model, save_model_artifacts
from .evaluate import compute_metrics, plot_confusion_matrices, plot_roc_curves, plot_performance_comparison
from .xai_explainer import WebGuardExplainer

__version__ = "1.0.0"
