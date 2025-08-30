"""
CatBoost Model Implementation for Bank Customer Churn Prediction
===============================================================

This module implements CatBoost models with different bootstrap types
following the 3rd place solution approach.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier

from .base import (
    BaseChurnModel, SalaryRounder, AgeRounder, FeatureGenerator, 
    Vectorizer, SVDRounder, CAT_FEATURES
)


class CatBoostChurnModel(BaseChurnModel):
    """
    CatBoost implementation for churn prediction with different bootstrap types.
    """
    
    def __init__(self, seed=42, n_splits=30, bootstrap_type='Ordered'):
        """
        Initialize CatBoost model.
        
        Args:
            seed (int): Random seed
            n_splits (int): Number of CV folds
            bootstrap_type (str): Bootstrap type ('Ordered', 'Bayesian', 'Bernoulli')
        """
        super().__init__(seed, n_splits)
        self.bootstrap_type = bootstrap_type
        self.model = None
        
    def get_default_params(self):
        """Get default CatBoost parameters from the 3rd place solution."""
        params = {
            'random_state': self.seed,
            'verbose': 0,
            'has_time': True,  # Important parameter from the winning solution
        }
        
        # Add bootstrap-specific parameters
        if self.bootstrap_type != 'Ordered':
            params['bootstrap_type'] = self.bootstrap_type
            
        return params
    
    def create_pipeline(self, params=None):
        """
        Create CatBoost pipeline following the 3rd place solution.
        
        Args:
            params (dict): CatBoost parameters
            
        Returns:
            sklearn.pipeline.Pipeline: Complete preprocessing + model pipeline
        """
        if params is None:
            params = self.get_default_params()
            
        # Define categorical features including SVD components
        cat_features = CAT_FEATURES + [f'SurnameSVD{i}' for i in range(4)] + [f'AllCatSVD{i}' for i in range(4)]
        params['cat_features'] = cat_features
        
        pipeline = make_pipeline(
            SalaryRounder,
            AgeRounder,
            FeatureGenerator,
            Vectorizer(
                cols=['Surname', 'AllCat'], 
                max_features=1000, 
                n_components=4
            ),
            SVDRounder,  # Important: round SVD features for better performance
            CatBoostClassifier(**params)
        )
        
        return pipeline
    
    def train_and_predict(self, train_df, orig_train_df, test_df, 
                         show_importance=False, label=None):
        """
        Train model and make predictions.
        
        Args:
            train_df: Training data
            orig_train_df: Original dataset
            test_df: Test data
            show_importance: Whether to show feature importance
            label: Model label for display
            
        Returns:
            Tuple of (validation_scores, oof_predictions, test_predictions)
        """
        if label is None:
            label = f"CatBoost_{self.bootstrap_type}"
        
        # Create model
        self.model = self.create_pipeline()
        
        print(f"🚀 Training {label} model...")
        val_scores, oof_preds, test_preds = self.cross_val_score(
            self.model, train_df, orig_train_df, test_df,
            label=label, show_importance=show_importance
        )
        
        return val_scores, oof_preds, test_preds


class CatBoostEnsemble:
    """
    Ensemble of different CatBoost configurations following the 3rd place solution.
    """
    
    def __init__(self, seed=42, n_splits=30):
        self.seed = seed
        self.n_splits = n_splits
        self.models = {}
        
    def create_variant_models(self):
        """
        Create different CatBoost model variants as used in the 3rd place solution.
        """
        
        # Standard CatBoost (Ordered bootstrap)
        self.models['cb_ordered'] = CatBoostChurnModel(
            seed=self.seed, 
            n_splits=self.n_splits, 
            bootstrap_type='Ordered'
        )
        
        # CatBoost with Bayesian bootstrap
        self.models['cb_bayesian'] = CatBoostChurnModel(
            seed=self.seed, 
            n_splits=self.n_splits, 
            bootstrap_type='Bayesian'
        )
        
        # CatBoost with Bernoulli bootstrap
        self.models['cb_bernoulli'] = CatBoostChurnModel(
            seed=self.seed, 
            n_splits=self.n_splits, 
            bootstrap_type='Bernoulli'
        )
        
        return self.models
    
    def train_ensemble(self, train_df, orig_train_df, test_df):
        """
        Train ensemble of CatBoost models with different bootstrap types.
        
        Args:
            train_df: Training data
            orig_train_df: Original dataset
            test_df: Test data
            
        Returns:
            dict: Results from all models
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}")
            print(f"{'='*50}")
            
            val_scores, oof_preds, test_preds = model.train_and_predict(
                train_df, orig_train_df, test_df, label=name
            )
            
            results[name] = {
                'val_scores': val_scores,
                'oof_predictions': oof_preds,
                'test_predictions': test_preds,
                'mean_score': np.mean(val_scores),
                'std_score': np.std(val_scores)
            }
            
        return results


class CatBoostAdvanced(CatBoostChurnModel):
    """
    Advanced CatBoost implementation with additional features.
    """
    
    def __init__(self, seed=42, n_splits=30, bootstrap_type='Ordered', 
                 use_eval_set=True, early_stopping_rounds=100):
        """
        Initialize advanced CatBoost model.
        
        Args:
            seed (int): Random seed
            n_splits (int): Number of CV folds
            bootstrap_type (str): Bootstrap type
            use_eval_set (bool): Whether to use evaluation set for early stopping
            early_stopping_rounds (int): Early stopping rounds
        """
        super().__init__(seed, n_splits, bootstrap_type)
        self.use_eval_set = use_eval_set
        self.early_stopping_rounds = early_stopping_rounds
        
    def get_advanced_params(self):
        """Get advanced CatBoost parameters."""
        params = self.get_default_params()
        
        # Add advanced parameters
        params.update({
            'iterations': 2000,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'border_count': 128,
            'thread_count': -1,
            'eval_metric': 'AUC',
        })
        
        if self.use_eval_set:
            params['early_stopping_rounds'] = self.early_stopping_rounds
            
        return params
    
    def create_pipeline(self, params=None):
        """Create advanced CatBoost pipeline."""
        if params is None:
            params = self.get_advanced_params()
            
        # Enhanced feature engineering for advanced model
        cat_features = CAT_FEATURES + [f'SurnameSVD{i}' for i in range(4)] + [f'AllCatSVD{i}' for i in range(4)]
        params['cat_features'] = cat_features
        
        pipeline = make_pipeline(
            SalaryRounder,
            AgeRounder,
            FeatureGenerator,
            Vectorizer(
                cols=['Surname', 'AllCat', 'EstimatedSalary', 'CustomerId'], 
                max_features=1500,  # More features for advanced model
                n_components=4
            ),
            SVDRounder,
            CatBoostClassifier(**params)
        )
        
        return pipeline


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing CatBoost implementations...")
    
    # Test different bootstrap types
    bootstrap_types = ['Ordered', 'Bayesian', 'Bernoulli']
    
    for bootstrap_type in bootstrap_types:
        model = CatBoostChurnModel(bootstrap_type=bootstrap_type)
        print(f"✅ {bootstrap_type} CatBoost model ready!")
    
    # Test ensemble
    ensemble = CatBoostEnsemble()
    ensemble.create_variant_models()
    print(f"✅ CatBoost ensemble with {len(ensemble.models)} models ready!")
    
    print("✅ CatBoost module ready for use!")