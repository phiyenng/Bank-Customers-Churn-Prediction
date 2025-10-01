# modules/models/xgboost.py
from xgboost import XGBClassifier
import joblib
from typing import Any, Dict, Optional, Tuple
import numpy as np


class XGBoostModel:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.model = XGBClassifier(**self.params)

    def fit(self, X, y, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None, sample_weight: Optional[np.ndarray] = None):
        if eval_set is not None:
            X_val, y_val = eval_set
            self.model.fit(X, y, eval_set=[(X_val, y_val)], sample_weight=sample_weight, verbose=False)
        else:
            self.model.fit(X, y, sample_weight=sample_weight)
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def get_native_model(self):
        """Return the underlying estimator for interpretability tools."""

        return self.model