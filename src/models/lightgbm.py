# modules/models/lightgbm.py
from lightgbm import LGBMClassifier
import joblib
from typing import Any, Dict, Optional, Tuple
import numpy as np


class LightGBMModel:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.model = LGBMClassifier(**self.params)

    def fit(self, X, y, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        if eval_set is not None:
            X_val, y_val = eval_set
            # Some LightGBM versions don't accept 'verbose' in fit; control via params instead
            self.model.fit(X, y, eval_set=[(X_val, y_val)])
        else:
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)