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

# Import from our custom modules
from .feature_engineering import Transformation
from .imbalance_handler import ImbalanceHandler

import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """
    Class to handle loading and combining multiple datasets.

    """
    
    def __init__(self, paths: dict):
        """
        Args:
            paths (dict): Dictionary from config.yaml, e.g.:
                {
                  "combine_sources": ["data/raw/train.csv", "data/raw/Churn_Modelling.csv"],
                  "test_path": "data/raw/test.csv",
                  "artifacts_dir": "saved_models"
                }
        """
        self.paths = paths
        self.original_df = None
        self.competition_train_df = None
        self.competition_test_df = None
        self.combined_train_df = None
    
    def load_original_dataset(self) -> pd.DataFrame:
        try:
            path = self.paths['combine_sources'][1]  # Churn_Modelling.csv
            self.original_df = pd.read_csv(path)
            return self.original_df
        except (FileNotFoundError, IndexError):
            self.original_df = None
            return None
            
    def load_competition_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            train_path = self.paths['combine_sources'][0]  # train.csv
            test_path = self.paths['test_path']
            self.competition_train_df = pd.read_csv(train_path)
            self.competition_test_df = pd.read_csv(test_path)
            return self.competition_train_df, self.competition_test_df
        except (FileNotFoundError, KeyError):
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
        self.original_df = self.original_df[common_cols]
        self.competition_train_df = self.competition_train_df[common_cols]
        
        # Concatenate: Original dataset - Competition dataset + Drop duplicates
        self.combined_train_df = pd.concat(
            [self.original_df, self.competition_train_df],
            ignore_index=True).drop_duplicates()
        
        return self.combined_train_df
    
    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all datasets and return them together.
        Returns:
            original_df, competition_test_df, combined_train_df
        """
        original = self.load_original_dataset()
        _, comp_test = self.load_competition_datasets()
        combined = self.combine_datasets()

        return original, comp_test, combined


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


