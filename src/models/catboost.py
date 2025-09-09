# modules/models/catboost.py
from catboost import CatBoostClassifier
import joblib
from typing import Any, Dict, Optional, Tuple
import numpy as np


class CatBoostModel:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {"verbose": 0}
        self.model = CatBoostClassifier(**self.params)

    def fit(self, X, y, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        if eval_set is not None:
            X_val, y_val = eval_set
            self.model.fit(X, y, eval_set=[(X_val, y_val)])
        else:
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)