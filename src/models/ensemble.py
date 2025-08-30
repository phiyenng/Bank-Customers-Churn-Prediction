"""
Ensemble Methods for Bank Customer Churn Prediction
===================================================

This module implements ensemble methods including voting, stacking, and
weighted combinations following the 3rd place solution approach.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import warnings
warnings.filterwarnings('ignore')

from .base import BaseChurnModel


class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Weighted ensemble following the 3rd place solution approach using Ridge regression.
    """
    
    def __init__(self, seed=42):
        """
        Initialize weighted ensemble.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed
        self.weights = None
        self.weight_model = None
        self.classes_ = None
        
    def fit(self, oof_predictions, y):
        """
        Fit the ensemble weights using Ridge regression.
        
        Args:
            oof_predictions (pd.DataFrame): Out-of-fold predictions from base models
            y (pd.Series): True labels
            
        Returns:
            self
        """
        self.classes_ = np.unique(y)
        
        # Use Ridge regression to find optimal weights
        self.weight_model = RidgeClassifier(random_state=self.seed)
        self.weight_model.fit(oof_predictions, y)
        
        # Extract and normalize weights
        self.weights = self.weight_model.coef_[0]
        self.weights = self.weights / self.weights.sum()
        
        # Create weights dataframe for display
        self.weights_df = pd.DataFrame(
            self.weights, 
            index=list(oof_predictions.columns), 
            columns=['weight']
        )
        
        return self
    
    def predict_proba(self, test_predictions):
        """
        Make weighted predictions.
        
        Args:
            test_predictions (pd.DataFrame): Test predictions from base models
            
        Returns:
            np.ndarray: Weighted predictions
        """
        weighted_preds = test_predictions.to_numpy() @ self.weights
        
        # Return probabilities in sklearn format
        proba = np.column_stack([1 - weighted_preds, weighted_preds])
        return proba
    
    def predict(self, test_predictions):
        """
        Make binary predictions.
        
        Args:
            test_predictions (pd.DataFrame): Test predictions from base models
            
        Returns:
            np.ndarray: Binary predictions
        """
        proba = self.predict_proba(test_predictions)
        return (proba[:, 1] > 0.5).astype(int)
    
    def get_weights(self):
        """
        Get model weights.
        
        Returns:
            pd.DataFrame: Model weights
        """
        return self.weights_df


class EnsembleManager:
    """
    Manager class for training and combining multiple models into ensembles.
    """
    
    def __init__(self, seed=42, n_splits=30):
        """
        Initialize ensemble manager.
        
        Args:
            seed (int): Random seed
            n_splits (int): Number of CV folds
        """
        self.seed = seed
        self.n_splits = n_splits
        self.models = {}
        self.oof_predictions = pd.DataFrame()
        self.test_predictions = pd.DataFrame()
        self.scores = {}
        
    def add_model_results(self, name, val_scores, oof_preds, test_preds):
        """
        Add results from a trained model.
        
        Args:
            name (str): Model name
            val_scores (list): Validation scores
            oof_preds (np.ndarray): Out-of-fold predictions
            test_preds (np.ndarray): Test predictions
        """
        self.oof_predictions[name] = oof_preds
        self.test_predictions[name] = test_preds
        self.scores[name] = {
            'mean': np.mean(val_scores),
            'std': np.std(val_scores),
            'scores': val_scores
        }
        
        print(f"✅ Added {name}: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f}")
    
    def create_weighted_ensemble(self, target):
        """
        Create weighted ensemble using Ridge regression.
        
        Args:
            target (pd.Series): True labels for training ensemble weights
            
        Returns:
            WeightedEnsemble: Fitted ensemble model
        """
        print("\n🎯 Creating weighted ensemble...")
        
        ensemble = WeightedEnsemble(seed=self.seed)
        ensemble.fit(self.oof_predictions, target)
        
        # Calculate ensemble score
        ensemble_preds = self.oof_predictions.to_numpy() @ ensemble.weights
        ensemble_score = roc_auc_score(target, ensemble_preds)
        
        print(f"🎉 Ensemble Score: {ensemble_score:.5f}")
        print("\n📊 Model Weights:")
        print(ensemble.get_weights().sort_values('weight', ascending=False))
        
        return ensemble
    
    def create_voting_ensemble(self, models_dict):
        """
        Create simple voting ensemble.
        
        Args:
            models_dict (dict): Dictionary of fitted models
            
        Returns:
            VotingClassifier: Fitted voting ensemble
        """
        print("\n🗳️  Creating voting ensemble...")
        
        # Create voting classifier
        estimators = [(name, model) for name, model in models_dict.items()]
        voting_ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        
        return voting_ensemble
    
    def create_stacking_ensemble(self, models_dict, meta_learner=None):
        """
        Create stacking ensemble.
        
        Args:
            models_dict (dict): Dictionary of base models
            meta_learner: Meta learner for stacking (default: RidgeClassifier)
            
        Returns:
            StackingClassifier: Fitted stacking ensemble
        """
        print("\n🏗️  Creating stacking ensemble...")
        
        if meta_learner is None:
            meta_learner = RidgeClassifier(random_state=self.seed)
        
        # Create stacking classifier
        estimators = [(name, model) for name, model in models_dict.items()]
        stacking_ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        )
        
        return stacking_ensemble
    
    def get_model_summary(self):
        """
        Get summary of all model performances.
        
        Returns:
            pd.DataFrame: Model performance summary
        """
        summary_data = []
        
        for name, score_info in self.scores.items():
            summary_data.append({
                'Model': name,
                'Mean_Score': score_info['mean'],
                'Std_Score': score_info['std'],
                'Score_Range': f"{score_info['mean']:.5f} ± {score_info['std']:.5f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Mean_Score', ascending=False)
        
        return summary_df
    
    def create_submission(self, ensemble_model, test_df, orig_test_combo, filename='submission.csv'):
        """
        Create submission file following the 3rd place solution format.
        
        Args:
            ensemble_model: Fitted ensemble model
            test_df: Test dataframe
            orig_test_combo: Original test combination for data leakage fixes
            filename: Output filename
            
        Returns:
            pd.DataFrame: Submission dataframe
        """
        print(f"\n📄 Creating submission file: {filename}")
        
        # Get ensemble predictions
        if hasattr(ensemble_model, 'weights'):
            # Weighted ensemble
            predictions = self.test_predictions.to_numpy() @ ensemble_model.weights
        else:
            # Other ensemble types
            predictions = ensemble_model.predict_proba(self.test_predictions)[:, 1]
        
        # Create submission following the 3rd place solution format
        submission = test_df.copy()
        
        # Apply data leakage fixes as in the original solution
        submission['Exited'] = np.where(
            orig_test_combo.Exited == 1, 0,
            np.where(orig_test_combo.Exited == 0, 1, predictions)
        )
        
        # Save submission
        submission['Exited'].to_csv(filename)
        
        print(f"✅ Submission saved: {filename}")
        print(f"📊 Prediction statistics:")
        print(f"   Mean: {submission['Exited'].mean():.4f}")
        print(f"   Std:  {submission['Exited'].std():.4f}")
        print(f"   Min:  {submission['Exited'].min():.4f}")
        print(f"   Max:  {submission['Exited'].max():.4f}")
        
        return submission


class ModelBlender:
    """
    Advanced model blending with different combination strategies.
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        
    def rank_average_blend(self, predictions_dict):
        """
        Blend predictions using rank averaging.
        
        Args:
            predictions_dict (dict): Dictionary of model predictions
            
        Returns:
            np.ndarray: Blended predictions
        """
        predictions_df = pd.DataFrame(predictions_dict)
        
        # Convert to ranks
        ranks_df = predictions_df.rank(pct=True)
        
        # Average ranks
        blended_ranks = ranks_df.mean(axis=1)
        
        return blended_ranks.values
    
    def power_average_blend(self, predictions_dict, powers=None):
        """
        Blend predictions using power averaging.
        
        Args:
            predictions_dict (dict): Dictionary of model predictions
            powers (dict): Power for each model (default: equal powers)
            
        Returns:
            np.ndarray: Blended predictions
        """
        if powers is None:
            powers = {name: 1.0 for name in predictions_dict.keys()}
        
        blended = np.zeros(len(list(predictions_dict.values())[0]))
        total_power = sum(powers.values())
        
        for name, preds in predictions_dict.items():
            weight = powers.get(name, 1.0) / total_power
            blended += weight * (preds ** powers.get(name, 1.0))
        
        return blended
    
    def geometric_mean_blend(self, predictions_dict):
        """
        Blend predictions using geometric mean.
        
        Args:
            predictions_dict (dict): Dictionary of model predictions
            
        Returns:
            np.ndarray: Blended predictions
        """
        predictions_array = np.column_stack(list(predictions_dict.values()))
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        predictions_array = np.clip(predictions_array, epsilon, 1 - epsilon)
        
        # Geometric mean
        log_mean = np.mean(np.log(predictions_array), axis=1)
        geometric_mean = np.exp(log_mean)
        
        return geometric_mean


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing ensemble implementations...")
    
    # Create dummy data for testing
    n_samples = 1000
    n_models = 5
    
    # Dummy OOF predictions
    oof_preds = pd.DataFrame({
        f'model_{i}': np.random.rand(n_samples) for i in range(n_models)
    })
    
    # Dummy target
    target = np.random.randint(0, 2, n_samples)
    
    # Test weighted ensemble
    ensemble = WeightedEnsemble()
    ensemble.fit(oof_preds, target)
    print("✅ Weighted ensemble created!")
    
    # Test ensemble manager
    manager = EnsembleManager()
    for i in range(n_models):
        manager.add_model_results(
            f'model_{i}',
            [0.8 + np.random.rand() * 0.1],  # Dummy scores
            np.random.rand(n_samples),       # Dummy OOF
            np.random.rand(500)              # Dummy test
        )
    
    print("✅ Ensemble manager tested!")
    
    # Test blender
    blender = ModelBlender()
    dummy_preds = {f'model_{i}': np.random.rand(100) for i in range(3)}
    blended = blender.rank_average_blend(dummy_preds)
    print("✅ Model blender tested!")
    
    print("✅ Ensemble module ready for use!")
