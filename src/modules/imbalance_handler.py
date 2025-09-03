"""
Imbalance Handling Module
=========================

This module provides tools to handle imbalanced datasets
using resampling techniques such as SMOTE, NearMiss,
SMOTEENN, and SMOTETomek.
"""

import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek


class ImbalanceHandler:
    """
    Handle class imbalance using oversampling, undersampling, or hybrid methods.
    """

    def __init__(self, method: str = "smote", random_state: int = 42):
        """
        Parameters
        ----------
        method : str
            Resampling method to apply. Options:
            - "smote"
            - "oversample"
            - "undersample"
            - "nearmiss"
            - "smoteenn"
            - "smotetomek"
            - "none"
        random_state : int
            Random state for reproducibility.
        """
        self.method = method
        self.random_state = random_state
        self.sampler = self._init_sampler()

    def _init_sampler(self):
        if self.method == "smote":
            return SMOTE(random_state=self.random_state)
        elif self.method == "oversample":
            return RandomOverSampler(random_state=self.random_state)
        elif self.method == "undersample":
            return RandomUnderSampler(random_state=self.random_state)
        elif self.method == "nearmiss":
            return NearMiss()
        elif self.method == "smoteenn":
            return SMOTEENN(random_state=self.random_state)
        elif self.method == "smotetomek":
            return SMOTETomek(random_state=self.random_state)
        elif self.method == "none":
            return None
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit_resample(self, X, y):
        """
        Resample the dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target labels

        Returns
        -------
        X_res, y_res : pd.DataFrame, pd.Series
            Resampled features and labels
        """
        if self.sampler is None:
            return X, y

        X_res, y_res = self.sampler.fit_resample(X, y)

        # Convert back to DataFrame/Series for consistency
        X_res = pd.DataFrame(X_res, columns=X.columns)
        y_res = pd.Series(y_res, name=y.name)

        return X_res, y_res
