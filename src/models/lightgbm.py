# modules/models/lightgbm.py
from lightgbm import LGBMClassifier
from .base_model import BaseModel

class LightGBMModel(BaseModel):
    """Wrapper for the LightGBM Classifier."""
    def _create_model(self):
        return LGBMClassifier(**self.params)