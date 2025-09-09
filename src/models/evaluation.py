from typing import Dict
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix


def get_evaluation_metrics(y_true, y_pred, y_proba, model_id: str) -> Dict[str, float]:
    return {
        'Model': model_id,
        'F1': classification_report(y_true, y_pred, output_dict=True)['weighted avg']['f1-score'],
        'Precision': classification_report(y_true, y_pred, output_dict=True)['weighted avg']['precision'],
        'Recall': classification_report(y_true, y_pred, output_dict=True)['weighted avg']['recall'],
        'ROC_AUC': roc_auc_score(y_true, y_proba),
        'PR_AUC': average_precision_score(y_true, y_proba)
    }


def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))


def compute_confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


