# modules/models/xgboost.py
from xgboost import XGBClassifier
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """Wrapper for the XGBoost Classifier."""
    def _create_model(self):
        return XGBClassifier(**self.params)