"""
Bank Customer Churn Prediction Pipeline
=======================================

Simple pipeline using custom modules and models from src/
"""

import sys
import warnings
import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('src')
warnings.filterwarnings('ignore')

# Import custom modules
from src.modules.data_processing import DataLoader, DataSplitter
from src.modules.imbalance_handler import ImbalanceHandler
from src.modules.evaluation import ModelEvaluator
from src.modules.feature_engineering import FeatureEngineeringPipeline
from src.models.xgboost import get_xgb_pipeline
from src.models.lightgbm import get_lgb_pipeline
from src.models.catboost import get_cb_pipeline


class ChurnPipeline:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.random_state = self.config.get('random_state', 42)
        
        # Initialize modules
        self.data_loader = DataLoader()
        self.data_splitter = DataSplitter(test_size=0.2, random_state=self.random_state)
        self.evaluator = ModelEvaluator(random_state=self.random_state)
        
        # Create output directories
        Path('results').mkdir(exist_ok=True)
        Path('artifacts').mkdir(exist_ok=True)
        Path('plots').mkdir(exist_ok=True)
        
        print(f"🚀 Pipeline ready with random_state={self.random_state}")
    
    def _load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_data(self):
        print("📊 Loading data...")
        df = self.data_loader.load_original_dataset()
        print(f"✅ Dataset loaded: {df.shape}")
        return df
    
    def prepare_and_split_data(self, df):
        print("🔧 Preparing and splitting data...")
        
        # Basic cleaning
        df = df.drop_duplicates()
        columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Use DataSplitter
        target_col = self.config['data']['target_column']
        X_train, X_test, y_train, y_test = self.data_splitter.split(df, target_col)
        
        print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"✅ Target distribution: {y_train.value_counts().to_dict()}")
        return X_train, X_test, y_train, y_test
    
    def preprocess_features(self, X_train, X_test, y_train):
        print("⚙️ Using FeatureEngineeringPipeline...")
        
        # Use custom feature engineering pipeline
        fe_pipeline = FeatureEngineeringPipeline(
            use_num=True, 
            num_method="yeo-johnson",
            use_cat=True, 
            cat_method="onehot"
        )
        
        # Fit and transform
        fe_pipeline.fit(X_train, y_train)
        X_train_processed = fe_pipeline.transform(X_train)
        X_test_processed = fe_pipeline.transform(X_test)
        
        # Save pipeline
        joblib.dump(fe_pipeline, 'artifacts/feature_pipeline.pkl')
        
        print(f"✅ Feature engineering done: {X_train_processed.shape}")
        return X_train_processed, X_test_processed
    
    def get_models(self):
        print("🔄 Loading models...")
        
        models = {}
        
        # XGBoost
        xgb_pipeline = get_xgb_pipeline()
        models['xgboost'] = xgb_pipeline.named_steps['xgbclassifier']
        
        # LightGBM
        lgb_pipeline = get_lgb_pipeline()
        models['lightgbm'] = lgb_pipeline.named_steps['lgbmclassifier']
        
        # CatBoost
        catboost_pipeline = get_cb_pipeline()
        models['catboost'] = catboost_pipeline.named_steps['catboostclassifier']
        
        print(f"✅ Models loaded: {list(models.keys())}")
        return models
    
    def handle_imbalance(self, X, y, method):
        if method == 'none':
            return X, y
        
        print(f"  ⚖️ Applying {method}...")
        handler = ImbalanceHandler(method=method, random_state=self.random_state)
        X_resampled, y_resampled = handler.fit_resample(X, y)
        
        print(f"    {X.shape} → {X_resampled.shape}")
        return X_resampled, y_resampled
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        print("🚀 Training models...")
        
        models = self.get_models()
        imbalance_methods = self.config['imbalance_methods']
        results = []
        trained_models = {}
        
        best_score = 0
        best_model = None
        best_combination = ""
        
        for method in imbalance_methods:
            print(f"\n--- {method.upper()} ---")
            X_balanced, y_balanced = self.handle_imbalance(X_train, y_train, method)
            
            for name, model in models.items():
                print(f"  🔄 Training {name}...")
                
                # Train
                model.fit(X_balanced, y_balanced)
                
                # Predict
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Use ModelEvaluator
                metrics = self.evaluator.evaluate_single_model(
                    y_test, y_pred_proba, y_pred, model_name=f"{name}_{method}"
                )
                roc_auc = metrics['roc_auc']
                f1 = metrics['f1_score']
                precision = metrics['precision']
                recall = metrics['recall']
                
                # Store results
                combination = f"{name}_{method}"
                result = {
                    'model': name,
                    'imbalance_method': method,
                    'combination': combination,
                    'roc_auc': roc_auc,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall
                }
                results.append(result)
                trained_models[combination] = model
                
                # Track best model
                if roc_auc > best_score:
                    best_score = roc_auc
                    best_model = model
                    best_combination = combination
                
                print(f"    ✅ ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}")
        
        return results, trained_models, best_model, best_combination, best_score
    
    def save_results(self, results, best_model, best_combination, best_score):
        print("💾 Saving results...")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv('results/pipeline_results.csv', index=False)
        
        # Save best model
        joblib.dump(best_model, 'artifacts/best_model.pkl')
        
        print(f"✅ Results saved")
        print(f"🏆 Best: {best_combination} (ROC-AUC: {best_score:.4f})")
        
        return results_df
    
    def create_plots(self, results_df, trained_models, X_test, y_test):
        print("📈 Creating plots...")
        
        # Performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # ROC-AUC by model
        sns.boxplot(data=results_df, x='model', y='roc_auc', ax=axes[0,0])
        axes[0,0].set_title('ROC-AUC by Model')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # F1-Score by model
        sns.boxplot(data=results_df, x='model', y='f1_score', ax=axes[0,1])
        axes[0,1].set_title('F1-Score by Model')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Precision vs Recall
        sns.scatterplot(data=results_df, x='precision', y='recall', 
                       hue='model', size='roc_auc', ax=axes[1,0])
        axes[1,0].set_title('Precision vs Recall')
        
        # Performance by imbalance method
        sns.boxplot(data=results_df, x='imbalance_method', y='roc_auc', ax=axes[1,1])
        axes[1,1].set_title('ROC-AUC by Imbalance Method')
        
        plt.tight_layout()
        plt.savefig('plots/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC curves for top models
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_5 = results_df.nlargest(5, 'roc_auc')
        for _, row in top_5.iterrows():
            model = trained_models[row['combination']]
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            ax.plot(fpr, tpr, label=f"{row['combination']} (AUC = {row['roc_auc']:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - Top 5 Models')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig('plots/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Plots saved")
    
    def run(self):
        print("🏦 BANK CUSTOMER CHURN PREDICTION PIPELINE")
        print("=" * 50)
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        df = self.load_data()
        
        # Prepare and split data
        X_train, X_test, y_train, y_test = self.prepare_and_split_data(df)
        
        # Preprocess
        X_train, X_test = self.preprocess_features(X_train, X_test, y_train)
        
        # Train and evaluate
        results, trained_models, best_model, best_combination, best_score = self.train_and_evaluate(
            X_train, X_test, y_train, y_test
        )
        
        # Save results
        results_df = self.save_results(results, best_model, best_combination, best_score)
        
        # Create plots
        self.create_plots(results_df, trained_models, X_test, y_test)
        
        # Summary
        print(f"\n📊 SUMMARY")
        print("=" * 30)
        print(f"🔢 Total models: {len(results)}")
        print(f"🏆 Best: {best_combination}")
        print(f"📈 Best ROC-AUC: {best_score:.4f}")
        
        top_5 = results_df.nlargest(5, 'roc_auc')[['combination', 'roc_auc', 'f1_score']]
        print(f"\n📋 Top 5:")
        print(top_5.round(4).to_string(index=False))
        
        print(f"\n⏰ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎉 Done!")


def main():
    pipeline = ChurnPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
