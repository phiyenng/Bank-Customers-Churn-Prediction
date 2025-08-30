"""
Data Processing Module
======================

This module handles data loading, combining datasets, and basic preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Class to handle loading and combining multiple datasets.

    """
    
    def __init__(self, data_path: str = "data/raw"):
        """
        Initialize DataLoader with path to raw data.
        """
        self.data_path = Path(data_path)
        self.original_df = None
        self.competition_train_df = None
        self.competition_test_df = None
        self.combined_train_df = None
        
    def load_original_dataset(self) -> pd.DataFrame:
        """
        Load the original Bank Customer Churn dataset (Churn_Modelling.csv).

        """
        try:
            file_path = self.data_path / "Churn_Modelling.csv"
            self.original_df = pd.read_csv(file_path)
            return self.original_df
        
        except FileNotFoundError:
            return None
            
    def load_competition_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load competition datasets (train.csv and test.csv).

        """
        try:
            train_path = self.data_path / "train.csv"
            test_path = self.data_path / "test.csv"
            
            self.competition_train_df = pd.read_csv(train_path)
            self.competition_test_df = pd.read_csv(test_path)
            return self.competition_train_df, self.competition_test_df
        
        except FileNotFoundError as e:
            return None, None
    
    def combine_datasets(self) -> pd.DataFrame:
        """
        Combine original and competition datasets using concatenation strategy.

        """
        if self.original_df is None:
            self.load_original_dataset()
        if self.competition_train_df is None:
            self.load_competition_datasets()
            
        # Remove unnecessary columns from original dataset if present
        if 'RowNumber' in self.original_df.columns:
            self.original_df = self.original_df.drop(['RowNumber'], axis=1)
        if 'CustomerId' in self.original_df.columns:
            self.original_df = self.original_df.drop(['CustomerId'], axis=1)
        if 'Surname' in self.original_df.columns:
            self.original_df = self.original_df.drop(['Surname'], axis=1)
            
        # Ensure columns are in the same order
        common_cols = list(set(self.original_df.columns) & set(self.competition_train_df.columns))
        
        # Reorder columns to match
        self.original_df = self.original_df[common_cols]
        self.competition_train_df = self.competition_train_df[common_cols]
        
        # Concatenate: Original dataset first, then competition dataset
        self.combined_train_df = pd.concat([
            self.original_df, 
            self.competition_train_df
        ], ignore_index=True)
        
        print(f"✅ Combined dataset shape: {self.combined_train_df.shape}")
        print(f"   - Original dataset: {len(self.original_df)} rows")
        print(f"   - Competition dataset: {len(self.competition_train_df)} rows")
        
        return self.combined_train_df
    
    def get_basic_info(self) -> dict:
        """
        Get basic information about all loaded datasets.
        
        Returns:
            dict: Information about datasets
        """
        info = {}
        
        if self.original_df is not None:
            info['original'] = {
                'shape': self.original_df.shape,
                'columns': list(self.original_df.columns),
                'missing_values': self.original_df.isnull().sum().sum(),
                'churn_rate': self.original_df['Exited'].mean() if 'Exited' in self.original_df.columns else 'N/A'
            }
            
        if self.competition_train_df is not None:
            info['competition_train'] = {
                'shape': self.competition_train_df.shape,
                'columns': list(self.competition_train_df.columns),
                'missing_values': self.competition_train_df.isnull().sum().sum(),
                'churn_rate': self.competition_train_df['Exited'].mean() if 'Exited' in self.competition_train_df.columns else 'N/A'
            }
            
        if self.competition_test_df is not None:
            info['competition_test'] = {
                'shape': self.competition_test_df.shape,
                'columns': list(self.competition_test_df.columns),
                'missing_values': self.competition_test_df.isnull().sum().sum(),
            }
            
        if self.combined_train_df is not None:
            info['combined'] = {
                'shape': self.combined_train_df.shape,
                'columns': list(self.combined_train_df.columns),
                'missing_values': self.combined_train_df.isnull().sum().sum(),
                'churn_rate': self.combined_train_df['Exited'].mean() if 'Exited' in self.combined_train_df.columns else 'N/A'
            }
            
        return info


class BasicPreprocessor:
    """Basic preprocessing utilities for the datasets."""
    
    @staticmethod
    def identify_feature_types(df: pd.DataFrame) -> dict:
        """
        Identify different types of features in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Dictionary containing different feature types
        """
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable if present
        if 'Exited' in numerical_features:
            numerical_features.remove('Exited')
            
        feature_types = {
            'numerical': numerical_features,
            'categorical': categorical_features,
            'target': 'Exited' if 'Exited' in df.columns else None
        }
        
        return feature_types
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> dict:
        """
        Check data quality issues.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Data quality report
        """
        quality_report = {
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns},
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        return quality_report
    
    @staticmethod
    def save_processed_data(df: pd.DataFrame, filename: str, output_path: str = "data/processed"):
        """
        Save processed data to specified path.
        
        Args:
            df (pd.DataFrame): Dataframe to save
            filename (str): Name of the output file
            output_path (str): Output directory path
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / filename
        df.to_csv(file_path, index=False)
        print(f"✅ Saved processed data: {file_path}")


def load_and_combine_data(data_path: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Convenience function to load and combine all datasets.
    
    Args:
        data_path (str): Path to raw data directory
        
    Returns:
        Tuple containing combined training data, test data, and info dictionary
    """
    loader = DataLoader(data_path)
    
    # Load all datasets
    loader.load_original_dataset()
    loader.load_competition_datasets()
    
    # Combine training data
    combined_train = loader.combine_datasets()
    
    # Get basic info
    info = loader.get_basic_info()
    
    return combined_train, loader.competition_test_df, info


if __name__ == "__main__":
    # Example usage
    print("=== Bank Customer Churn Data Processing ===")
    
    # Load and combine data
    train_df, test_df, info = load_and_combine_data()
    
    # Display basic information
    print("\n=== Dataset Information ===")
    for dataset_name, dataset_info in info.items():
        print(f"\n{dataset_name.upper()}:")
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")
    
    # Check feature types
    feature_types = BasicPreprocessor.identify_feature_types(train_df)
    print(f"\n=== Feature Types ===")
    for feature_type, features in feature_types.items():
        print(f"{feature_type}: {features}")
    
    # Check data quality
    quality_report = BasicPreprocessor.check_data_quality(train_df)
    print(f"\n=== Data Quality Report ===")
    print(f"Missing values: {sum(quality_report['missing_values'].values())}")
    print(f"Duplicate rows: {quality_report['duplicate_rows']}")
    print(f"Memory usage: {quality_report['memory_usage']:.2f} MB")