"""
LightGBM Model Implementation for Bank Customer Churn Prediction
===============================================================

This module implements LightGBM models with hyperparameter optimization
following the 3rd place solution approach.
"""

import numpy as np
import pandas as pd
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna not available. Install with: pip install optuna")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from category_encoders import CatBoostEncoder, MEstimateEncoder

from .base import (
    BaseChurnModel, SalaryRounder, AgeRounder, FeatureGenerator, 
    Vectorizer, CAT_FEATURES
)


class LightGBMChurnModel(BaseChurnModel):
    """
    LightGBM implementation for churn prediction with Optuna optimization.
    """
    
    def __init__(self, seed=42, n_splits=30, use_optuna=True, n_trials=100):
        """
        Initialize LightGBM model.
        
        Args:
            seed (int): Random seed
            n_splits (int): Number of CV folds
            use_optuna (bool): Whether to use Optuna for hyperparameter tuning
            n_trials (int): Number of Optuna trials
        """
        super().__init__(seed, n_splits)
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.n_trials = n_trials
        self.best_params = None
        self.model = None
        
    def get_default_params(self):
        """Get default LightGBM parameters from the 3rd place solution."""
        return {
            'learning_rate': 0.01864960338160943,
            'max_depth': 9,
            'subsample': 0.6876252164703066,
            'min_child_weight': 0.8117588782708633,
            'reg_lambda': 6.479178739677389,
            'reg_alpha': 3.2952573115561234,
            'n_estimators': 1000,
            'random_state': self.seed,
            'verbose': -1
        }
    
    def create_pipeline(self, params=None):
        """
        Create LightGBM pipeline following the 3rd place solution.
        
        Args:
            params (dict): LightGBM parameters
            
        Returns:
            sklearn.pipeline.Pipeline: Complete preprocessing + model pipeline
        """
        if params is None:
            params = self.get_default_params()
            
        pipeline = make_pipeline(
            SalaryRounder,
            AgeRounder,
            FeatureGenerator,
            Vectorizer(
                cols=['Surname', 'AllCat'], 
                max_features=1000, 
                n_components=3
            ),
            CatBoostEncoder(cols=['Surname', 'AllCat', 'CreditScore', 'Age']),
            MEstimateEncoder(cols=['Geography', 'Gender', 'NumOfProducts']),
            StandardScaler(),
            LGBMClassifier(**params)
        )
        
        return pipeline
    
    def objective(self, trial, train_df, orig_train_df, test_df):
        """
        Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            train_df: Training data
            orig_train_df: Original dataset
            test_df: Test data
            
        Returns:
            float: Mean validation score
        """
        params = {
            'learning_rate': trial.suggest_float('learning_rate', .001, .1, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'subsample': trial.suggest_float('subsample', .5, 1),
            'min_child_weight': trial.suggest_float('min_child_weight', .1, 15, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', .1, 20, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', .1, 10, log=True),
            'n_estimators': 1000,
            'random_state': self.seed,
            'verbose': -1
            # Note: boosting_type can be 'dart' for additional regularization
        }
        
        optuna_model = self.create_pipeline(params)
        optuna_score, _, _ = self.cross_val_score(
            optuna_model, train_df, orig_train_df, test_df
        )
        
        return np.mean(optuna_score)
    
    def optimize_hyperparameters(self, train_df, orig_train_df, test_df):
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            train_df: Training data
            orig_train_df: Original dataset  
            test_df: Test data
            
        Returns:
            dict: Best parameters found
        """
        if not OPTUNA_AVAILABLE:
            print("⚠️  Optuna not available. Using default parameters.")
            return self.get_default_params()
            
        print("🔍 Starting LightGBM hyperparameter optimization...")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self.objective(trial, train_df, orig_train_df, test_df),
            n_trials=self.n_trials
        )
        
        self.best_params = study.best_params.copy()
        self.best_params.update({
            'n_estimators': 1000,
            'random_state': self.seed,
            'verbose': -1
        })
        
        print(f"✅ Best LightGBM parameters found:")
        for key, value in self.best_params.items():
            print(f"   {key}: {value}")
            
        return self.best_params
    
    def train_and_predict(self, train_df, orig_train_df, test_df, 
                         show_importance=False, label="LightGBM"):
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
        # Use optimized parameters if available, otherwise use defaults
        if self.use_optuna and self.best_params is None:
            self.optimize_hyperparameters(train_df, orig_train_df, test_df)
            params = self.best_params
        elif self.best_params is not None:
            params = self.best_params
        else:
            params = self.get_default_params()
        
        # Create and train model
        self.model = self.create_pipeline(params)
        
        print(f"🚀 Training {label} model...")
        val_scores, oof_preds, test_preds = self.cross_val_score(
            self.model, train_df, orig_train_df, test_df,
            label=label, show_importance=show_importance
        )
        
        return val_scores, oof_preds, test_preds


class LightGBMEnsemble:
    """
    Ensemble of different LightGBM configurations.
    """
    
    def __init__(self, seed=42, n_splits=30):
        self.seed = seed
        self.n_splits = n_splits
        self.models = {}
        
    def create_variant_models(self):
        """Create different LightGBM model variants."""
        
        # Standard LightGBM
        self.models['lgb_standard'] = LightGBMChurnModel(
            seed=self.seed, n_splits=self.n_splits, use_optuna=False
        )
        
        # LightGBM with DART boosting
        lgb_dart = LightGBMChurnModel(
            seed=self.seed, n_splits=self.n_splits, use_optuna=False
        )
        dart_params = lgb_dart.get_default_params()
        dart_params['boosting_type'] = 'dart'
        lgb_dart.best_params = dart_params
        self.models['lgb_dart'] = lgb_dart
        
        return self.models
    
    def train_ensemble(self, train_df, orig_train_df, test_df):
        """
        Train ensemble of LightGBM models.
        
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


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing LightGBM implementation...")
    
    # This would be called with actual data
    # model = LightGBMChurnModel(use_optuna=False)
    # val_scores, oof_preds, test_preds = model.train_and_predict(train_df, orig_train_df, test_df)
    
    print("✅ LightGBM module ready for use!")