# modules/evaluation.py
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

def get_evaluation_metrics(y_true, y_pred, y_pred_proba, model_name: str) -> dict:
    """
    Calculates a dictionary of standard classification metrics.
    """
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'ROC_AUC': roc_auc_score(y_true, y_pred_proba),
        'F1_Score': f1_score(y_true, y_pred, average='weighted'),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted')
    }
    return metrics

def print_classification_report(y_true, y_pred):
    """
    Prints a detailed classification report.
    """
    print(classification_report(y_true, y_pred, digits=4))