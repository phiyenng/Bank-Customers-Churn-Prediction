"""
Imbalance Handler Module
========================

This module handles imbalanced datasets using various sampling techniques
and class weights. Supports oversampling, undersampling, hybrid methods,
and class weight adjustment.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek

import warnings
warnings.filterwarnings('ignore')


class ImbalanceHandler:
    """
    Class to handle imbalanced datasets using various sampling techniques
    and class weights. Supports oversampling, undersampling, hybrid methods,
    and class weight adjustment.
    """

    def __init__(self, method: str = 'smote', random_state: int = 42):
        """
        Initialize the ImbalanceHandler.

        Parameters
        ----------
        method : str, default='smote'
            Options: 'smote', 'adasyn', 'nearmiss', 'smote_enn', 'smote_tomek', 'class_weight'
        random_state : int, default=42
            Random state for reproducibility
        """
        self.method = method
        self.random_state = random_state
        self.sampler = None
        self.fitted = False

    def _get_sampler(self, sampling_strategy: str = 'auto'):
        """
        Returns the appropriate sampler object based on self.method.
        """

        if self.method == 'class_weight':
            return None

        sampler_map = {
            'smote': lambda: SMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state, k_neighbors=5),
            'adasyn': lambda: ADASYN(sampling_strategy=sampling_strategy, random_state=self.random_state, n_neighbors=5),
            'nearmiss': lambda: NearMiss(sampling_strategy=sampling_strategy, version=1),
            'smote_enn': lambda: SMOTEENN(sampling_strategy=sampling_strategy, random_state=self.random_state),
            'smote_tomek': lambda: SMOTETomek(sampling_strategy=sampling_strategy, random_state=self.random_state)
        }

        if self.method not in sampler_map:
            raise ValueError(f"Unknown method: {self.method}")

        return sampler_map[self.method]()

    def fit_resample(self, X: pd.DataFrame, y: pd.Series, sampling_strategy: str = 'auto') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit the sampler and resample the data.

        Returns
        -------
        X_resampled, y_resampled : Tuple[pd.DataFrame, pd.Series]
        """
        if self.method == 'class_weight':
            print("Warning: method='class_weight' does not resample data. Use get_class_weights() for model training.")
            self.fitted = True
            return X.copy(), y.copy()

        self.sampler = self._get_sampler(sampling_strategy)
        X_res, y_res = self.sampler.fit_resample(X, y)

        # Convert back to pandas, reset index to avoid misalignment
        X_res = pd.DataFrame(X_res, columns=X.columns)
        y_res = pd.Series(y_res, name=y.name)

        self.fitted = True
        return X_res, y_res

    def get_class_weights(self, y: pd.Series, method: str = 'balanced') -> dict:
        """
        Compute class weights for imbalanced datasets.
        """

        unique_classes = np.unique(y)

        if method in ['balanced', 'balanced_subsample']:
            weights = compute_class_weight(method, classes=unique_classes, y=y)
        elif method == 'custom':
            class_counts = y.value_counts()
            total_samples = len(y)
            weights = [total_samples / (len(unique_classes) * class_counts[cls]) for cls in unique_classes]
        else:
            raise ValueError(f"Unknown method: {method}")

        return dict(zip(unique_classes, weights))

    def get_sampling_info(self, y: pd.Series) -> dict:
        """
        Get class distribution and imbalance information.
        """
        class_counts = y.value_counts().sort_index()
        total_samples = len(y)
        class_ratios = class_counts / total_samples

        return {
            'class_counts': class_counts.to_dict(),
            'class_ratios': class_ratios.to_dict(),
            'total_samples': total_samples,
            'n_classes': len(class_counts),
            'imbalance_ratio': class_counts.max() / class_counts.min(),
            'majority_class': class_counts.idxmax(),
            'minority_class': class_counts.idxmin()
        }

    def plot_class_distribution(self, y_before: pd.Series, y_after: Optional[pd.Series] = None, title: str = "Class Distribution") -> None:
        """
        Plot class distribution before and optionally after resampling.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        counts_before = y_before.value_counts().sort_index()
        ax.bar(counts_before.index.astype(str), counts_before.values, alpha=0.6, label='Before')

        if y_after is not None:
            counts_after = y_after.value_counts().sort_index()
            ax.bar(counts_after.index.astype(str), counts_after.values, alpha=0.6, label='After')

        ax.set_title(title)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.legend()

        # Add counts on bars
        for i, count in enumerate(counts_before.values):
            ax.text(i, count + 0.01 * counts_before.max(), str(count), ha='center', va='bottom')
        if y_after is not None:
            for i, count in enumerate(counts_after.values):
                ax.text(i, count + 0.01 * counts_after.max(), str(count), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def get_sampler_params(self) -> dict:
        if not self.fitted or self.sampler is None:
            return {}
        return {
            'method': self.method,
            'random_state': self.random_state,
            'sampler_params': self.sampler.get_params() if hasattr(self.sampler, 'get_params') else {}
        }

    def set_sampler_params(self, params: dict) -> 'ImbalanceHandler':
        if 'method' in params:
            self.method = params['method']
        if 'random_state' in params:
            self.random_state = params['random_state']
        self.fitted = True
        return self
