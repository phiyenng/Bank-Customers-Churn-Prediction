"""
Base Model Classes for Bank Customer Churn Prediction
====================================================

This module contains base classes and utilities for model training,
cross-validation, and evaluation following the 3rd place solution approach.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from category_encoders import CatBoostEncoder, MEstimateEncoder
import warnings
warnings.filterwarnings('ignore')


class BaseChurnModel:
    """
    Base class for churn prediction models following the 3rd place solution approach.
    """
    
    def __init__(self, seed=42, n_splits=30):
        """
        Initialize base model with cross-validation settings.
        
        Args:
            seed (int): Random seed for reproducibility
            n_splits (int): Number of CV folds
        """
        self.seed = seed
        self.n_splits = n_splits
        self.skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
        
    def cross_val_score(self, estimator, train_df, orig_train_df, test_df, 
                       label='', include_original=True, show_importance=False, 
                       add_reverse=False):
        """
        Perform cross-validation following the 3rd place solution strategy.
        
        Args:
            estimator: ML model pipeline
            train_df: Competition training data
            orig_train_df: Original dataset
            test_df: Test data
            label: Model label for display
            include_original: Whether to include original dataset in training
            show_importance: Whether to show feature importance
            add_reverse: Whether to add reversed data for augmentation
            
        Returns:
            Tuple of (validation_scores, oof_predictions, test_predictions)
        """
        X = train_df.copy()
        y = X.pop('Exited')
        
        # Create combo datasets for data leakage prevention
        orig_comp_combo = train_df.merge(orig_train_df, on=list(test_df), how='left')
        orig_comp_combo.index = train_df.index
        
        orig_test_combo = test_df.merge(orig_train_df, on=list(test_df), how='left')
        orig_test_combo.index = test_df.index
        
        # Initialize prediction arrays and score lists
        val_predictions = np.zeros((len(X)))
        train_scores, val_scores = [], []
        
        feature_importances_table = pd.DataFrame({'value': 0}, index=list(X.columns))
        test_predictions = np.zeros((len(test_df)))
        
        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(X, y)):
            
            model = clone(estimator)
            
            # Define train set
            X_train = X.iloc[train_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            
            # Define validation set
            X_val = X.iloc[val_idx].reset_index(drop=True)
            y_val = y.iloc[val_idx].reset_index(drop=True)
            
            # Add original dataset if specified
            if include_original:
                X_train = pd.concat([orig_train_df.drop('Exited', axis=1), X_train]).reset_index(drop=True)
                y_train = pd.concat([orig_train_df.Exited, y_train]).reset_index(drop=True)
                
            # Add reversed data for augmentation if specified
            if add_reverse:
                X_train = pd.concat([X_train, X_train.iloc[::-1]]).reset_index(drop=True)
                y_train = pd.concat([y_train, y_train.iloc[::-1]]).reset_index(drop=True)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_preds = model.predict_proba(X_train)[:, 1]
            val_preds = model.predict_proba(X_val)[:, 1]
                      
            val_predictions[val_idx] += val_preds
            test_predictions += model.predict_proba(test_df)[:, 1] / self.skf.get_n_splits()
            
            # Calculate feature importance if requested
            if show_importance:
                feature_importances_table['value'] += permutation_importance(
                    model, X_val, y_val, random_state=self.seed, 
                    scoring=make_scorer(roc_auc_score, needs_proba=True), 
                    n_repeats=5
                ).importances_mean / self.skf.get_n_splits()
            
            # Evaluate model for this fold
            train_score = roc_auc_score(y_train, train_preds)
            val_score = roc_auc_score(y_val, val_preds)
            
            # Append scores
            train_scores.append(train_score)
            val_scores.append(val_score)
       
        # Display results
        if show_importance:
            plt.figure(figsize=(20, 30))
            plt.title(f'Features with Biggest Importance of {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} Model', 
                     size=25, weight='bold')
            sns.barplot(feature_importances_table.sort_values('value', ascending=False).T, 
                       orient='h', palette='viridis')
            plt.show()
        else:
            print(f'Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | '
                  f'Train Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')
            
        # Apply data leakage fixes from original solution
        val_predictions = np.where(orig_comp_combo.Exited_y == 1, 0, 
                                 np.where(orig_comp_combo.Exited_y == 0, 1, val_predictions))
        test_predictions = np.where(orig_test_combo.Exited == 1, 0, 
                                  np.where(orig_test_combo.Exited == 0, 1, test_predictions))
        
        return val_scores, val_predictions, test_predictions


class CustomTransformers:
    """
    Collection of custom transformers from the 3rd place solution.
    """
    
    @staticmethod
    def nullify(x):
        """Replace 0 balance with NaN."""
        x_copy = x.copy()
        x_copy['Balance'] = x_copy['Balance'].replace({0: np.nan})
        return x_copy

    @staticmethod
    def salary_rounder(x):
        """Round salary values for better encoding."""
        x_copy = x.copy()
        x_copy['EstimatedSalary'] = (x_copy['EstimatedSalary'] * 100).astype(np.uint64)
        return x_copy

    @staticmethod
    def age_rounder(x):
        """Round age values for better encoding."""
        x_copy = x.copy()
        x_copy['Age'] = (x_copy['Age'] * 10).astype(np.uint16)
        return x_copy

    @staticmethod
    def balance_rounder(x):
        """Round balance values for better encoding."""
        x_copy = x.copy()
        x_copy['Balance'] = (x_copy['Balance'] * 100).astype(np.uint64)
        return x_copy

    @staticmethod
    def feature_generator(x):
        """Generate additional features from the 3rd place solution."""
        x_copy = x.copy()
        
        # Key engineered features from the winning solution
        x_copy['IsActive_by_CreditCard'] = x_copy['HasCrCard'] * x_copy['IsActiveMember']
        x_copy['Products_Per_Tenure'] = x_copy['Tenure'] / x_copy['NumOfProducts']
        x_copy['ZeroBalance'] = (x_copy['Balance'] == 0).astype(np.uint8)
        x_copy['AgeCat'] = np.round(x_copy.Age/20).astype(np.uint16)
        
        # Complex categorical feature combination
        x_copy['AllCat'] = (x_copy['Surname'] + x_copy['Geography'] + x_copy['Gender'] + 
                           x_copy.EstimatedSalary.astype('str') + x_copy.CreditScore.astype('str') + 
                           x_copy.Age.astype('str') + x_copy.NumOfProducts.astype('str') + 
                           x_copy.Tenure.astype('str') + x_copy.CustomerId.astype('str'))
        
        return x_copy

    @staticmethod
    def svd_rounder(x):
        """Round SVD components for better performance."""
        x_copy = x.copy()
        for col in [column for column in list(x) if 'SVD' in column]:
            x_copy[col] = (x_copy[col] * 1e18).astype(np.int64)
        return x_copy


class FeatureDropper(BaseEstimator, TransformerMixin):
    """Custom transformer to drop specified columns."""
    
    def __init__(self, cols):
        self.cols = cols
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        return x.drop(self.cols, axis=1)


class Categorizer(BaseEstimator, TransformerMixin):
    """Custom transformer to convert columns to categorical type."""
    
    def __init__(self, cols: list):
        self.cols = cols
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        return x.astype({cat: 'category' for cat in self.cols})


class Vectorizer(BaseEstimator, TransformerMixin):
    """
    Custom vectorizer using TF-IDF and SVD following the 3rd place solution.
    """
    
    def __init__(self, max_features=1000, cols=['Surname'], n_components=3, seed=42):
        self.max_features = max_features
        self.cols = cols
        self.n_components = n_components
        self.seed = seed
        
    def fit(self, x, y=None):
        self.vectorizer_dict = {}
        self.decomposer_dict = {}
        
        for col in self.cols:
            self.vectorizer_dict[col] = TfidfVectorizer(max_features=self.max_features).fit(x[col].astype(str), y)
            self.decomposer_dict[col] = TruncatedSVD(
                random_state=self.seed, 
                n_components=self.n_components
            ).fit(self.vectorizer_dict[col].transform(x[col].astype(str)), y)
        
        return self
    
    def transform(self, x):
        vectorized = {}
        
        for col in self.cols:
            vectorized[col] = self.vectorizer_dict[col].transform(x[col].astype(str))
            vectorized[col] = self.decomposer_dict[col].transform(vectorized[col])
        
        vectorized_df = pd.concat([
            pd.DataFrame(vectorized[col]).rename({
                i: f'{col}SVD{i}' for i in range(self.n_components)
            }, axis=1) for col in self.cols
        ], axis=1)
        
        return pd.concat([x.reset_index(drop=True), vectorized_df], axis=1)


# Create function transformers for easy pipeline integration
Nullify = FunctionTransformer(CustomTransformers.nullify)
SalaryRounder = FunctionTransformer(CustomTransformers.salary_rounder)
AgeRounder = FunctionTransformer(CustomTransformers.age_rounder)
BalanceRounder = FunctionTransformer(CustomTransformers.balance_rounder)
FeatureGenerator = FunctionTransformer(CustomTransformers.feature_generator)
SVDRounder = FunctionTransformer(CustomTransformers.svd_rounder)


# Define categorical features as used in the 3rd place solution
CAT_FEATURES = [
    'CustomerId', 'Surname', 'EstimatedSalary', 'Geography', 'Gender', 
    'Tenure', 'Age', 'NumOfProducts', 'IsActiveMember', 'CreditScore', 
    'AllCat', 'IsActive_by_CreditCard'
]
