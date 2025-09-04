# modules/models/catboost.py
from catboost import CatBoostClassifier
from .base_model import BaseModel

class CatBoostModel(BaseModel):
    """Wrapper for the CatBoost Classifier."""
    def _create_model(self):
        return CatBoostClassifier(**self.params)