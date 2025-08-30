"""
Main Training Script - 3rd Place Solution Implementation
=======================================================

This script implements the complete training pipeline following the
3rd place solution approach for Bank Customer Churn Prediction.
"""

import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our models
from src.models import (
    XGBoostChurnModel, XGBoostEnsemble,
    LightGBMChurnModel, LightGBMEnsemble, 
    CatBoostChurnModel, CatBoostEnsemble,
    EnsembleManager, WeightedEnsemble,
    TENSORFLOW_AVAILABLE
)

if TENSORFLOW_AVAILABLE:
    from src.models import TensorFlowChurnModel

# Import data processing
from src.data_processing import load_and_combine_data


class ChurnPredictionPipeline:
    """
    Complete training pipeline following the 3rd place solution.
    """
    
    def __init__(self, seed=42, n_splits=30, use_optuna=False):
        """
        Initialize the training pipeline.
        
        Args:
            seed (int): Random seed for reproducibility
            n_splits (int): Number of cross-validation folds
            use_optuna (bool): Whether to use Optuna for hyperparameter optimization
        """
        self.seed = seed
        self.n_splits = n_splits
        self.use_optuna = use_optuna
        self.ensemble_manager = EnsembleManager(seed=seed, n_splits=n_splits)
        
        # Data containers
        self.train_df = None
        self.orig_train_df = None
        self.test_df = None
        
    def load_data(self, data_path="data/raw"):
        """
        Load and prepare data.
        
        Args:
            data_path (str): Path to data directory
        """
        print("📊 Loading datasets...")
        
        # Load datasets directly
        churn_path = os.path.join(data_path, "Churn_Modelling.csv")
        train_path = os.path.join(data_path, "train.csv") 
        test_path = os.path.join(data_path, "test.csv")
        
        # Load original dataset
        self.orig_train_df = pd.read_csv(churn_path)
        
        # Clean original dataset
        if 'RowNumber' in self.orig_train_df.columns:
            self.orig_train_df = self.orig_train_df.drop(['RowNumber'], axis=1)
        if 'CustomerId' in self.orig_train_df.columns:
            self.orig_train_df = self.orig_train_df.drop(['CustomerId'], axis=1)
        if 'Surname' in self.orig_train_df.columns:
            self.orig_train_df = self.orig_train_df.drop(['Surname'], axis=1)
        
        # Load competition datasets
        self.train_df = pd.read_csv(train_path, index_col='id').astype({
            'IsActiveMember': np.uint8, 
            'HasCrCard': np.uint8
        })
        self.test_df = pd.read_csv(test_path, index_col='id').astype({
            'IsActiveMember': np.uint8, 
            'HasCrCard': np.uint8
        })
        
        print(f"✅ Data loaded successfully!")
        print(f"   Original dataset: {self.orig_train_df.shape}")
        print(f"   Competition train: {self.train_df.shape}")
        print(f"   Competition test: {self.test_df.shape}")
        
    def train_individual_models(self):
        """Train individual models following the 3rd place solution."""
        
        print("\n" + "="*60)
        print("🚀 TRAINING INDIVIDUAL MODELS")
        print("="*60)
        
        # 1. Train XGBoost
        print("\n🔸 Training XGBoost...")
        xgb_model = XGBoostChurnModel(
            seed=self.seed, 
            n_splits=self.n_splits, 
            use_optuna=self.use_optuna
        )
        val_scores, oof_preds, test_preds = xgb_model.train_and_predict(
            self.train_df, self.orig_train_df, self.test_df, label="XGBoost"
        )
        self.ensemble_manager.add_model_results("XGB", val_scores, oof_preds, test_preds)
        
        # 2. Train LightGBM
        print("\n🔸 Training LightGBM...")
        lgb_model = LightGBMChurnModel(
            seed=self.seed, 
            n_splits=self.n_splits, 
            use_optuna=self.use_optuna
        )
        val_scores, oof_preds, test_preds = lgb_model.train_and_predict(
            self.train_df, self.orig_train_df, self.test_df, label="LightGBM"
        )
        self.ensemble_manager.add_model_results("LGB", val_scores, oof_preds, test_preds)
        
        # 3. Train CatBoost variants
        print("\n🔸 Training CatBoost variants...")
        catboost_ensemble = CatBoostEnsemble(seed=self.seed, n_splits=self.n_splits)
        catboost_ensemble.create_variant_models()
        catboost_results = catboost_ensemble.train_ensemble(
            self.train_df, self.orig_train_df, self.test_df
        )
        
        # Add CatBoost results to ensemble manager
        for name, results in catboost_results.items():
            self.ensemble_manager.add_model_results(
                name, results['val_scores'], 
                results['oof_predictions'], results['test_predictions']
            )
        
        # 4. Train TensorFlow (if available)
        if TENSORFLOW_AVAILABLE:
            print("\n🔸 Training TensorFlow...")
            tf_model = TensorFlowChurnModel(seed=self.seed, n_splits=self.n_splits)
            val_scores, oof_preds, test_preds = tf_model.train_and_predict(
                self.train_df, self.orig_train_df, self.test_df, label="TensorFlow"
            )
            self.ensemble_manager.add_model_results("TF", val_scores, oof_preds, test_preds)
        else:
            print("\n⚠️  TensorFlow not available, skipping...")
    
    def create_ensemble(self):
        """Create weighted ensemble following the 3rd place solution."""
        
        print("\n" + "="*60)
        print("🎯 CREATING ENSEMBLE")
        print("="*60)
        
        # Get target for ensemble training
        target = self.train_df['Exited']
        
        # Create weighted ensemble
        self.weighted_ensemble = self.ensemble_manager.create_weighted_ensemble(target)
        
        return self.weighted_ensemble
    
    def generate_submission(self, filename="submission.csv"):
        """Generate submission file."""
        
        print("\n" + "="*60)
        print("📄 GENERATING SUBMISSION")
        print("="*60)
        
        # Create data leakage prevention combo (following original solution)
        orig_test_combo = self.test_df.merge(
            self.orig_train_df, on=list(self.test_df.columns), how='left'
        )
        orig_test_combo.index = self.test_df.index
        
        # Create submission
        submission = self.ensemble_manager.create_submission(
            self.weighted_ensemble, self.test_df, orig_test_combo, filename
        )
        
        return submission
    
    def plot_results(self):
        """Plot training results and analysis."""
        
        print("\n" + "="*60)
        print("📊 PLOTTING RESULTS") 
        print("="*60)
        
        # Model performance summary
        summary_df = self.ensemble_manager.get_model_summary()
        
        # Plot model performance
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Model Scores
        plt.subplot(2, 2, 1)
        plt.barh(summary_df['Model'], summary_df['Mean_Score'], color='skyblue')
        plt.xlabel('ROC AUC Score')
        plt.title('Model Performance Comparison')
        plt.grid(axis='x', alpha=0.3)
        
        # Subplot 2: Model Weights
        plt.subplot(2, 2, 2)
        weights_df = self.weighted_ensemble.get_weights()
        plt.pie(weights_df['weight'], labels=weights_df.index, autopct='%1.1f%%')
        plt.title('Ensemble Model Weights')
        
        # Subplot 3: Prediction Distribution
        plt.subplot(2, 2, 3)
        ensemble_test_preds = (self.ensemble_manager.test_predictions.to_numpy() @ 
                              self.weighted_ensemble.weights)
        plt.hist(ensemble_test_preds, bins=50, alpha=0.7, color='lightcoral')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Frequency')
        plt.title('Test Predictions Distribution')
        
        # Subplot 4: OOF Predictions vs Target
        plt.subplot(2, 2, 4)
        ensemble_oof_preds = (self.ensemble_manager.oof_predictions.to_numpy() @ 
                             self.weighted_ensemble.weights)
        target = self.train_df['Exited']
        
        # Create ROC-like plot
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(target, ensemble_oof_preds)
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Ensemble OOF')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print("\n📋 FINAL RESULTS SUMMARY:")
        print("="*40)
        print(summary_df.to_string(index=False))
        
        # Print ensemble score
        ensemble_oof_preds = (self.ensemble_manager.oof_predictions.to_numpy() @ 
                             self.weighted_ensemble.weights)
        from sklearn.metrics import roc_auc_score
        ensemble_score = roc_auc_score(target, ensemble_oof_preds)
        print(f"\n🏆 Final Ensemble Score: {ensemble_score:.5f}")
    
    def run_complete_pipeline(self, data_path="data/raw", submission_filename="submission.csv"):
        """
        Run the complete training pipeline.
        
        Args:
            data_path (str): Path to data directory
            submission_filename (str): Name for submission file
        """
        print("🚀 STARTING COMPLETE TRAINING PIPELINE")
        print("Following 3rd Place Solution Approach")
        print("="*60)
        
        # Step 1: Load data
        self.load_data(data_path)
        
        # Step 2: Train individual models
        self.train_individual_models()
        
        # Step 3: Create ensemble
        self.create_ensemble()
        
        # Step 4: Generate submission
        self.generate_submission(submission_filename)
        
        # Step 5: Plot results
        self.plot_results()
        
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📄 Submission saved as: {submission_filename}")
        print("🏆 Ready for Kaggle submission!")


def main():
    """Main execution function."""
    
    # Configuration
    SEED = 42
    N_SPLITS = 30
    USE_OPTUNA = False  # Set to True for hyperparameter optimization (takes longer)
    DATA_PATH = "data/raw"
    SUBMISSION_FILE = "submission_3rd_place_solution.csv"
    
    # Initialize and run pipeline
    pipeline = ChurnPredictionPipeline(
        seed=SEED, 
        n_splits=N_SPLITS, 
        use_optuna=USE_OPTUNA
    )
    
    try:
        pipeline.run_complete_pipeline(
            data_path=DATA_PATH,
            submission_filename=SUBMISSION_FILE
        )
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
