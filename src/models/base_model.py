# modules/models/base_model.py
from abc import ABC, abstractmethod
import joblib

class BaseModel(ABC):
    """
    Abstract Base Class for all models.
    It defines a common interface for fitting, predicting, and saving/loading.
    """
    def __init__(self, params: dict = None):
        self.params = params if params is not None else {}
        self.model = self._create_model()

    @abstractmethod
    def _create_model(self):
        """Initializes the specific model instance."""
        pass

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, file_path: str):
        joblib.dump(self.model, file_path)

    def load(self, file_path: str):
        self.model = joblib.load(file_path)
        return self