"""Model evaluation utilities for metrics reporting and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


def get_evaluation_metrics(y_true, y_pred, y_proba, model_id: str) -> Dict[str, float]:
    """Compute core evaluation metrics for a binary classifier."""

    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "Model": model_id,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": report["weighted avg"]["f1-score"],
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "ROC_AUC": roc_auc_score(y_true, y_proba),
        "PR_AUC": average_precision_score(y_true, y_proba),
    }


def print_classification_report(y_true, y_pred) -> None:
    """Print the sklearn classification report for quick inspection."""

    print(classification_report(y_true, y_pred))


def compute_confusion(y_true, y_pred):
    """Return the confusion matrix for the provided predictions."""

    return confusion_matrix(y_true, y_pred)


def _default_class_names(class_names: Optional[Sequence[str]], n_classes: int) -> Sequence[str]:
    if class_names is not None:
        return class_names
    return [str(idx) for idx in range(n_classes)]


def _annotate_matrix(ax, matrix: np.ndarray) -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black")


def plot_confusion_matrix(
    y_true,
    y_pred,
    output_path: Path,
    model_id: str,
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = True,
) -> None:
    """Save a confusion matrix plot to ``output_path``."""

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    labels = _default_class_names(class_names, cm.shape[0])
    fig, ax = plt.subplots(figsize=(6, 5))
    heatmap = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(heatmap, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(f"Confusion Matrix - {model_id}")

    _annotate_matrix(ax, cm)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_roc_curve(
    y_true,
    y_proba,
    output_path: Path,
    model_id: str,
) -> None:
    """Save an ROC curve plot for the provided probabilities."""

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})", color="tab:blue")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_id}")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_precision_recall_curve(
    y_true,
    y_proba,
    output_path: Path,
    model_id: str,
) -> None:
    """Save a precision-recall curve plot for the provided probabilities."""

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="tab:green", label=f"PR curve (AP = {pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {model_id}")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_evaluation_plots(
    y_true,
    y_pred,
    y_proba,
    output_dir: Path,
    model_id: str,
    class_names: Optional[Sequence[str]] = None,
) -> None:
    """
    Generate confusion matrix, ROC, and PR curve plots and save them to ``output_dir``.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        output_path=output_dir / f"{model_id}_confusion_matrix.png",
        model_id=model_id,
        class_names=class_names,
        normalize=True,
    )

    plot_roc_curve(
        y_true=y_true,
        y_proba=y_proba,
        output_path=output_dir / f"{model_id}_roc_curve.png",
        model_id=model_id,
    )

    plot_precision_recall_curve(
        y_true=y_true,
        y_proba=y_proba,
        output_path=output_dir / f"{model_id}_pr_curve.png",
        model_id=model_id,
    )


def generate_shap_plots(
    model,
    X,
    output_dir: Path,
    model_id: str,
    sample_size: int = 2000,
    max_display: int = 20,
) -> None:
    """Generate SHAP summary and bar plots for feature importance interpretation."""

    if shap is None:
        print("SHAP library is not installed. Skipping interpretability plots.")
        return

    if X is None or len(X) == 0:
        print(f"No data available to compute SHAP values for {model_id}.")
        return

    if not hasattr(model, "predict"):
        print(f"Model object for {model_id} does not expose predict(). Skipping SHAP plots.")
        return

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X.copy()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"Could not compute SHAP values for {model_id}: {exc}")
        return

    shap_output_dir = output_dir / "shap"
    shap_output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(shap_values, list):
        # For binary classification shap returns values per class; use the positive class
        shap_to_use = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_to_use = shap_values

    # Summary (beeswarm) plot
    shap.summary_plot(
        shap_to_use,
        X_sample,
        max_display=max_display,
        show=False,
        plot_size=(8, 6),
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(shap_output_dir / f"{model_id}_shap_summary.png", dpi=300)
    plt.close(fig)

    # Bar plot
    shap.summary_plot(
        shap_to_use,
        X_sample,
        max_display=max_display,
        plot_type="bar",
        show=False,
        plot_size=(8, 6),
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(shap_output_dir / f"{model_id}_shap_bar.png", dpi=300)
    plt.close(fig)

