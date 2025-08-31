"""
Data Processing Module
======================

This module handles data loading, combining datasets, and basic preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

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
        self.original_df = self.original_df.drop(['RowNumber'], axis=1, errors='ignore')

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