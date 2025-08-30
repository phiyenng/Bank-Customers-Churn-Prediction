"""
XGBoost Model Implementation for Bank Customer Churn Prediction
==============================================================

This module implements XGBoost models with hyperparameter optimization
following the 3rd place solution approach.
"""

import numpy as np
import pandas as pd
import optuna
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from category_encoders import CatBoostEncoder, MEstimateEncoder

from .base import (
    BaseChurnModel, SalaryRounder, AgeRounder, FeatureGenerator, 
    Vectorizer, CAT_FEATURES
)


class XGBoostChurnModel(BaseChurnModel):
    """
    XGBoost implementation for churn prediction with Optuna optimization.
    """
    
    def __init__(self, seed=42, n_splits=30, use_optuna=True, n_trials=50):
        """
        Initialize XGBoost model.
        
        Args:
            seed (int): Random seed
            n_splits (int): Number of CV folds
            use_optuna (bool): Whether to use Optuna for hyperparameter tuning
            n_trials (int): Number of Optuna trials
        """
        super().__init__(seed, n_splits)
        self.use_optuna = use_optuna
        self.n_trials = n_trials
        self.best_params = None
        self.model = None
        
    def get_default_params(self):
        """Get default XGBoost parameters from the 3rd place solution."""
        return {
            'eta': 0.04007938900538817,
            'max_depth': 5,
            'subsample': 0.8858539721226424,
            'colsample_bytree': 0.41689519430449395,
            'min_child_weight': 0.4225662401139526,
            'reg_lambda': 1.7610231110037127,
            'reg_alpha': 1.993860687732973,
            'n_estimators': 1000,
            'random_state': self.seed,
            'tree_method': 'hist'
        }
    
    def create_pipeline(self, params=None):
        """
        Create XGBoost pipeline following the 3rd place solution.
        
        Args:
            params (dict): XGBoost parameters
            
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
                cols=['Surname', 'AllCat', 'EstimatedSalary', 'CustomerId'], 
                max_features=1000, 
                n_components=3
            ),
            CatBoostEncoder(cols=['CustomerId', 'Surname', 'EstimatedSalary', 'AllCat', 'CreditScore']),
            MEstimateEncoder(cols=['Geography', 'Gender']),
            XGBClassifier(**params)
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
            'eta': trial.suggest_float('eta', .001, .3, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 30),
            'subsample': trial.suggest_float('subsample', .5, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', .1, 1),
            'min_child_weight': trial.suggest_float('min_child_weight', .1, 20, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', .01, 20, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', .01, 10, log=True),
            'n_estimators': 1000,
            'random_state': self.seed,
            'tree_method': 'hist',
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
        print("🔍 Starting XGBoost hyperparameter optimization...")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self.objective(trial, train_df, orig_train_df, test_df),
            n_trials=self.n_trials
        )
        
        self.best_params = study.best_params.copy()
        self.best_params.update({
            'n_estimators': 1000,
            'random_state': self.seed,
            'tree_method': 'hist'
        })
        
        print(f"✅ Best XGBoost parameters found:")
        for key, value in self.best_params.items():
            print(f"   {key}: {value}")
            
        return self.best_params
    
    def train_and_predict(self, train_df, orig_train_df, test_df, 
                         show_importance=False, label="XGBoost"):
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


class XGBoostEnsemble:
    """
    Ensemble of different XGBoost configurations.
    """
    
    def __init__(self, seed=42, n_splits=30):
        self.seed = seed
        self.n_splits = n_splits
        self.models = {}
        
    def create_variant_models(self):
        """Create different XGBoost model variants."""
        
        # Standard XGBoost
        self.models['xgb_standard'] = XGBoostChurnModel(
            seed=self.seed, n_splits=self.n_splits, use_optuna=False
        )
        
        # XGBoost with different feature engineering
        self.models['xgb_enhanced'] = XGBoostChurnModel(
            seed=self.seed, n_splits=self.n_splits, use_optuna=False
        )
        
        return self.models
    
    def train_ensemble(self, train_df, orig_train_df, test_df):
        """
        Train ensemble of XGBoost models.
        
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
    print("🧪 Testing XGBoost implementation...")
    
    # This would be called with actual data
    # model = XGBoostChurnModel(use_optuna=False)
    # val_scores, oof_preds, test_preds = model.train_and_predict(train_df, orig_train_df, test_df)
    
    print("✅ XGBoost module ready for use!")