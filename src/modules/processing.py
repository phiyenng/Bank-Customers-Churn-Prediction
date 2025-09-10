"""
Data Processing Module
======================

This module handles data loading, combining datasets, and basic preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer, StandardScaler, QuantileTransformer, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek

import warnings
warnings.filterwarnings('ignore')

# ==== CONFIGURATION ====
DATA_PATH = Path('data/raw')
PROCESSED_PATH = Path('data/processed')
ORIGINAL_DATASET_PATH = DATA_PATH / 'Churn_Modelling.csv'
COMPETITION_TRAIN_PATH = DATA_PATH / 'train.csv'
COMPETITION_TEST_PATH = DATA_PATH / 'test.csv'


class DataLoader:
    """
    Class to handle loading and combining multiple datasets.

    """
    
    def __init__(self):
        self.original_df = None
        self.competition_train_df = None
        self.competition_test_df = None
        self.combined_train_df = None
    
    def load_original_dataset(self) -> pd.DataFrame:
        try:
            self.original_df = pd.read_csv(ORIGINAL_DATASET_PATH)
            return self.original_df
        
        except FileNotFoundError:
            self.original_df = None
            return None
            
    def load_competition_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            self.competition_train_df = pd.read_csv(COMPETITION_TRAIN_PATH)
            self.competition_test_df = pd.read_csv(COMPETITION_TEST_PATH)
            return self.competition_train_df, self.competition_test_df
        
        except FileNotFoundError as e:
            self.competition_train_df, self.competition_test_df = None, None
            return None, None
    
    def combine_datasets(self) -> pd.DataFrame:
        """
        Combine original and competition datasets.

        """
        if self.original_df is None:
            self.original_df = self.load_original_dataset()
        if self.competition_train_df is None:
            self.competition_train_df, self.competition_test_df = self.load_competition_datasets()

        if self.original_df is None or self.competition_train_df is None:
            return None
            
        # Remove unnecessary columns from original dataset
        self.original_df = self.original_df.drop(['RowNumber','CustomerId','Surname'], axis=1, errors='ignore')

        # Ensure columns are in the same order
        common_cols = list(set(self.original_df.columns) & set(self.competition_train_df.columns))
        
        # Reorder columns to match
        self.original_df = self.original_df[common_cols]
        self.competition_train_df = self.competition_train_df[common_cols]
        
        # Concatenate: Original dataset - Competition dataset
        self.combined_train_df = pd.concat([
            self.original_df, 
            self.competition_train_df
        ], ignore_index=True)
        
        return self.combined_train_df


class OutlierHandler:
    """
    Class to handle outliers using IQR (Interquartile Range) method.
    Automatically skips binary columns (0/1) and target variable if provided.
    """
    
    def __init__(self, iqr_multiplier: float = 1.5, target_col: Optional[str] = None):
        """
        Initialize the OutlierHandler.
        
        Parameters:
        -----------
        iqr_multiplier : float, default=1.5
            Multiplier for IQR to determine outlier bounds.
        target_col : str, optional
            Name of target column to exclude from outlier detection.
        """
        self.iqr_multiplier = iqr_multiplier
        self.target_col = target_col
        self.outlier_bounds = {}
        self.outlier_counts = {}
        
    def _get_numeric_columns(self, df: pd.DataFrame, columns: Optional[List[str]]) -> List[str]:
        """
        Helper to get valid numeric columns excluding binary and target.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        valid_cols = []
        for col in columns:
            if self.target_col and col == self.target_col:
                continue
            if df[col].nunique() <= 2:  # skip binary
                continue
            valid_cols.append(col)
        
        return valid_cols

    def detect_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detect outliers in specified numerical columns using IQR method.
        """
        columns = self._get_numeric_columns(df, columns)
        outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - (self.iqr_multiplier * IQR)
            upper_bound = Q3 + (self.iqr_multiplier * IQR)
            
            self.outlier_bounds[col] = {'lower': lower_bound, 'upper': upper_bound}
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_mask[col] = outliers
            self.outlier_counts[col] = outliers.sum()
                
        return outlier_mask
    
    def remove_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None, 
                       strategy: str = 'any', max_iter: int = 10, verbose: bool = True) -> pd.DataFrame:
        """
        Remove outliers iteratively from the dataframe until none remain or max_iter is reached.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe.
        columns : list of str, optional
            Columns to check. If None, use all numeric columns (excluding binary + target).
        strategy : str, default='any'
            'any' -> remove row if any column has an outlier
            'all' -> remove row only if all columns are outliers
        max_iter : int, default=10
            Maximum number of iterations.
        verbose : bool, default=True
            Whether to print progress per iteration.
        """
        cleaned_df = df.copy()
        columns = self._get_numeric_columns(cleaned_df, columns)
        
        for i in range(max_iter):
            # Detect outliers
            outlier_mask = self.detect_outliers(cleaned_df, columns)
            total_outliers = outlier_mask.sum().sum()
            
            if verbose:
                print(f"Iteration {i+1}: {total_outliers} outliers")
            
            # Stop if no outliers left
            if total_outliers == 0:
                break
            
            # Remove rows with outliers
            if strategy == 'any':
                rows_to_remove = outlier_mask.any(axis=1)
            elif strategy == 'all':
                rows_to_remove = outlier_mask.all(axis=1)
            else:
                raise ValueError("Strategy must be 'any' or 'all'")
            
            cleaned_df = cleaned_df[~rows_to_remove].copy()
        
        return cleaned_df
    
    def cap_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Cap outliers instead of removing them.
        """
        columns = self._get_numeric_columns(df, columns)
        df_capped = df.copy()
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - (self.iqr_multiplier * IQR)
            upper_bound = Q3 + (self.iqr_multiplier * IQR)
            
            df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_capped
    
    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of detected outliers.
        """
        if not self.outlier_bounds:
            return pd.DataFrame()
        
        summary_data = []
        for col, bounds in self.outlier_bounds.items():
            summary_data.append({
                'column': col,
                'outlier_count': self.outlier_counts.get(col, 0),
                'lower_bound': bounds['lower'],
                'upper_bound': bounds['upper'],
                'iqr_multiplier': self.iqr_multiplier
            })
        
        return pd.DataFrame(summary_data)


class Transformation:
    """
    Optimized feature-wise transformations for Gradient Boosting models.
    Includes optional OneHot encoding for categorical columns.
    """
    
    def __init__(self, method: str = 'standard', handle_categorical: bool = True):
        # 'method' retained for backward compatibility; not used in optimized mapping
        self.method = method
        self.handle_categorical = handle_categorical
        self.encoder = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.fitted = False
        
    def fit(self, df: pd.DataFrame, numeric: Optional[List[str]] = None, categorical: Optional[List[str]] = None):
        """
        Fit transformation parameters on training data.
        """
        if numeric is None:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if categorical is None and self.handle_categorical:
            categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
        else:
            categorical = []

        self.numeric_cols = numeric
        self.categorical_cols = categorical

        # Optimized per-feature transformers for gradient boosting models
        self.optimized_transformers = {}
        for col in numeric:
            if col == 'Age':
                # PowerTransformer (yeo-johnson) + StandardScaler
                pt = PowerTransformer(method='yeo-johnson', standardize=False)
                ss = StandardScaler()
                self.optimized_transformers[col] = ('power_standard', (pt, ss))
                pt.fit(df[[col]])
                ss.fit(pt.transform(df[[col]]))

            elif col == 'Balance':
                # QuantileTransformer to uniform
                qt = QuantileTransformer(n_quantiles=min(1000, len(df)), output_distribution='uniform', random_state=42)
                self.optimized_transformers[col] = ('quantile_uniform', (qt,))
                qt.fit(df[[col]])

            elif col == 'CreditScore':
                # Box-Cox (requires positive), shift if needed, then StandardScaler
                shift = 0.0
                series = df[col]
                if (series <= 0).any():
                    shift = abs(series.min()) + 1e-6
                pt = PowerTransformer(method='box-cox', standardize=False)
                ss = StandardScaler()
                self.optimized_transformers[col] = ('boxcox_standard', (pt, ss, shift))
                pt.fit((series + shift).values.reshape(-1,1))
                ss.fit(pt.transform((series + shift).values.reshape(-1,1)))

            elif col == 'EstimatedSalary':
                # QuantileTransformer (uniform) + StandardScaler
                qt = QuantileTransformer(n_quantiles=min(1000, len(df)), output_distribution='uniform', random_state=42)
                ss = StandardScaler()
                self.optimized_transformers[col] = ('quantile_standard', (qt, ss))
                qt.fit(df[[col]])
                ss.fit(qt.transform(df[[col]]))

            elif col == 'Tenure':
                # MinMaxScaler to range (-2, 2)
                mm = MinMaxScaler(feature_range=(-2, 2))
                self.optimized_transformers[col] = ('minmax_custom', (mm,))
                mm.fit(df[[col]])

            elif col == 'NumOfProducts':
                # MinMaxScaler + StandardScaler
                mm = MinMaxScaler()
                ss = StandardScaler()
                self.optimized_transformers[col] = ('minmax_standard', (mm, ss))
                mm.fit(df[[col]])
                ss.fit(mm.transform(df[[col]]))

            elif col == 'HasCrCard':
                # MinMaxScaler to (-1, 1)
                mm = MinMaxScaler(feature_range=(-1, 1))
                self.optimized_transformers[col] = ('minmax_neg1_1', (mm,))
                mm.fit(df[[col]])

            elif col == 'IsActiveMember':
                # Preserve as-is (important negative correlation per request)
                self.optimized_transformers[col] = ('none', tuple())
            else:
                # Pass-through for unspecified columns
                self.optimized_transformers[col] = ('none', tuple())

        # Fit encoder for categorical
        if self.handle_categorical and categorical:
            self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown="ignore")
            self.encoder.fit(df[categorical])

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to the dataframe.
        """
        if not self.fitted:
            raise ValueError("Call fit() before transform().")

        df_transformed = pd.DataFrame(index=df.index)

        # --- Numeric scaling ---
        for col in self.numeric_cols:
            if col not in df.columns: 
                continue

            if col in getattr(self, 'optimized_transformers', {}):
                mode, objs = self.optimized_transformers[col]

                if mode == 'power_standard':
                    pt, ss = objs
                    val = pt.transform(df[[col]])
                    df_transformed[col] = ss.transform(val)

                elif mode == 'quantile_uniform':
                    (qt,) = objs
                    df_transformed[col] = qt.transform(df[[col]])

                elif mode == 'boxcox_standard':
                    pt, ss, shift = objs
                    arr = (df[col] + shift).values.reshape(-1,1)
                    val = pt.transform(arr)
                    df_transformed[col] = ss.transform(val)

                elif mode == 'quantile_standard':
                    qt, ss = objs
                    val = qt.transform(df[[col]])
                    df_transformed[col] = ss.transform(val)

                elif mode == 'minmax_custom':
                    (mm,) = objs
                    df_transformed[col] = mm.transform(df[[col]])

                elif mode == 'minmax_standard':
                    mm, ss = objs
                    val = mm.transform(df[[col]])
                    df_transformed[col] = ss.transform(val)

                elif mode == 'minmax_neg1_1':
                    (mm,) = objs
                    df_transformed[col] = mm.transform(df[[col]])

                elif mode == 'none':
                    df_transformed[col] = df[col]
                else:
                    df_transformed[col] = df[col]
            else:
                # fallback already set to 'none' in fit
                df_transformed[col] = df[col]

        # --- Categorical encoding ---
        if self.handle_categorical and self.categorical_cols:
            encoded = self.encoder.transform(df[self.categorical_cols])
            encoded_df = pd.DataFrame(encoded, 
                                      columns=self.encoder.get_feature_names_out(self.categorical_cols),
                                      index=df.index)
            df_transformed = pd.concat([df_transformed, encoded_df], axis=1)

        return df_transformed

    def fit_transform(self, df: pd.DataFrame, numeric: Optional[List[str]] = None, categorical: Optional[List[str]] = None) -> pd.DataFrame:
        return self.fit(df, numeric, categorical).transform(df)


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
