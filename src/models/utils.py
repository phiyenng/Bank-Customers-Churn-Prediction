import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import roc_auc_score

def cross_val_score(model, X=None, y=None, folds=5, show_importance=False):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    preds = cross_val_predict(model, X, y, cv=kf, method="predict_proba")
    scores = []
    for train_idx, val_idx in kf.split(X):
        val_preds = preds[val_idx, 1]
        score = roc_auc_score(y[val_idx], val_preds)
        scores.append(score)
    return np.mean(scores), preds, None
