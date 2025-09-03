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
        print("📊 Loading combined dataset...")
        
        # Use combined dataset first
        df = self.data_loader.combine_datasets()
        if df is not None:
            print(f"✅ Combined dataset loaded: {df.shape}")
            return df
        
        # Fallback to original dataset
        df = self.data_loader.load_original_dataset()
        print(f"✅ Original dataset loaded: {df.shape}")
        return df
    
    def prepare_and_split_data(self, df):
        print("🔧 Step 1: Clean and split combined dataset for objectivity...")
        
        # Basic cleaning only
        df = df.drop_duplicates()
        columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # IMMEDIATELY split for objectivity
        target_col = self.config['data']['target_column']
        X_train, X_test, y_train, y_test = self.data_splitter.split(df, target_col)
        
        print(f"✅ Split completed - Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"✅ Training target distribution: {y_train.value_counts().to_dict()}")
        return X_train, X_test, y_train, y_test
    
    def process_features(self, X_train, X_test, y_train):
        print("⚙️ Step 2: Execute complex feature processing pipeline...")
        print("   → Creating new features, encoding, transformations")
        
        # Use custom feature engineering pipeline with advanced options
        fe_pipeline = FeatureEngineeringPipeline(
            use_num=True, 
            num_method="yeo-johnson",
            use_cat=True, 
            cat_method="onehot",
            use_arithmetic=True,  # Create new features
            use_reduction=False,  # Disable PCA for now
            reduction_method="pca",
            n_components=5
        )
        
        # FIT on training set only to avoid data leakage
        print("   → Fitting processors on training set only...")
        fe_pipeline.fit(X_train, y_train)
        
        # Transform both sets using fitted processors
        print("   → Transforming training set...")
        X_train_processed = fe_pipeline.transform(X_train)
        
        print("   → Transforming test set with same fitted processors...")
        X_test_processed = fe_pipeline.transform(X_test)
        
        # Save fitted pipeline for future use
        joblib.dump(fe_pipeline, 'artifacts/fitted_feature_pipeline.pkl')
        
        print(f"✅ Complex feature processing completed: {X_train_processed.shape}")
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
    
    def apply_smote_and_train(self, X_train_processed, y_train):
        print("⚖️ Step 3: Apply SMOTE on processed training data...")
        
        # Apply SMOTE to resolve imbalance
        handler = ImbalanceHandler(method='smote', random_state=self.random_state)
        X_train_balanced, y_train_balanced = handler.fit_resample(X_train_processed, y_train)
        
        print(f"   → Original: {X_train_processed.shape}")
        print(f"   → Balanced: {X_train_balanced.shape}")
        print(f"   → Class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
        
        return X_train_balanced, y_train_balanced
    
    def train_models(self, X_train_balanced, y_train_balanced):
        print("🚀 Step 4: Train XGBoost, LightGBM, and CatBoost on balanced data...")
        
        models = self.get_models()
        trained_models = {}
        
        # Convert to numpy arrays if needed for compatibility
        if hasattr(X_train_balanced, 'values'):
            X_train_array = X_train_balanced.values
        else:
            X_train_array = X_train_balanced
            
        if hasattr(y_train_balanced, 'values'):
            y_train_array = y_train_balanced.values
        else:
            y_train_array = y_train_balanced
        
        for name, model in models.items():
            print(f"   → Training {name.upper()}...")
            model.fit(X_train_array, y_train_array)
            trained_models[name] = model
            print(f"   ✅ {name.upper()} training completed")
        
        return trained_models
    
    def evaluate_on_original_test(self, trained_models, X_test_processed, y_test):
        print("📊 Step 5: Evaluate on original test set for reliable results...")
        
        results = []
        best_score = 0
        best_model = None
        best_model_name = ""
        
        # Convert to numpy array if needed
        if hasattr(X_test_processed, 'values'):
            X_test_array = X_test_processed.values
        else:
            X_test_array = X_test_processed
        
        for name, model in trained_models.items():
            print(f"   → Evaluating {name.upper()}...")
            
            # Predict on original test set
            y_pred = model.predict(X_test_array)
            y_pred_proba = model.predict_proba(X_test_array)[:, 1]
            
            # Use ModelEvaluator for comprehensive metrics
            metrics = self.evaluator.evaluate_single_model(
                y_test, y_pred_proba, y_pred, model_name=name
            )
            
            result = {
                'model': name,
                'roc_auc': metrics['roc_auc'],
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            }
            results.append(result)
            
            # Track best model
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = model
                best_model_name = name
            
            print(f"   ✅ {name.upper()}: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1_score']:.4f}")
        
        return results, trained_models, best_model, best_model_name, best_score
    
    def save_results(self, results, best_model, best_model_name, best_score):
        print("💾 Saving results...")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv('results/final_pipeline_results.csv', index=False)
        
        # Save best model and fitted processors
        joblib.dump(best_model, 'artifacts/best_final_model.pkl')
        
        print(f"✅ Results saved to results/final_pipeline_results.csv")
        print(f"✅ Best model saved to artifacts/best_final_model.pkl")
        print(f"🏆 Best model: {best_model_name} (ROC-AUC: {best_score:.4f})")
        
        return results_df
    
    def create_plots(self, results_df, trained_models, X_test, y_test):
        print("📈 Creating performance visualizations...")
        
        # Convert to numpy array if needed
        if hasattr(X_test, 'values'):
            X_test_array = X_test.values
        else:
            X_test_array = X_test
        
        # Performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Final Pipeline - Model Performance on Original Test Set', fontsize=16, fontweight='bold')
        
        # ROC-AUC comparison
        sns.barplot(data=results_df, x='model', y='roc_auc', ax=axes[0,0])
        axes[0,0].set_title('ROC-AUC by Model')
        axes[0,0].set_ylim(0, 1)
        
        # F1-Score comparison  
        sns.barplot(data=results_df, x='model', y='f1_score', ax=axes[0,1])
        axes[0,1].set_title('F1-Score by Model')
        axes[0,1].set_ylim(0, 1)
        
        # Precision vs Recall
        sns.scatterplot(data=results_df, x='precision', y='recall', 
                       hue='model', size='roc_auc', sizes=(100, 300), ax=axes[1,0])
        axes[1,0].set_title('Precision vs Recall')
        axes[1,0].set_xlim(0, 1)
        axes[1,0].set_ylim(0, 1)
        
        # Metrics comparison
        metrics_melted = results_df.melt(id_vars=['model'], 
                                       value_vars=['roc_auc', 'f1_score', 'precision', 'recall'],
                                       var_name='metric', value_name='score')
        sns.barplot(data=metrics_melted, x='model', y='score', hue='metric', ax=axes[1,1])
        axes[1,1].set_title('All Metrics Comparison')
        axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('plots/final_pipeline_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC curves
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for _, row in results_df.iterrows():
            model = trained_models[row['model']]
            y_pred_proba = model.predict_proba(X_test_array)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            ax.plot(fpr, tpr, label=f"{row['model']} (AUC = {row['roc_auc']:.3f})", linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate') 
        ax.set_title('ROC Curves - Final Pipeline Results')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig('plots/final_pipeline_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualizations saved to plots/ directory")
    
    def run(self):
        print("🏦 BANK CUSTOMER CHURN PREDICTION PIPELINE")
        print("=" * 60)
        print("📋 OBJECTIVE MACHINE LEARNING WORKFLOW:")
        print("   1️⃣  Load combined dataset")
        print("   2️⃣  Immediately split for objectivity")
        print("   3️⃣  Complex feature processing (fit on train only)")
        print("   4️⃣  Apply SMOTE on processed training data")
        print("   5️⃣  Train models on balanced data")
        print("   6️⃣  Evaluate on original test set")
        print("=" * 60)
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load combined dataset
        df = self.load_data()
        
        # Step 2: Immediately split for objectivity
        X_train, X_test, y_train, y_test = self.prepare_and_split_data(df)
        
        # Step 3: Complex feature processing (fit on train only)
        X_train_processed, X_test_processed = self.process_features(X_train, X_test, y_train)
        
        # Step 4: Apply SMOTE on processed training data
        X_train_balanced, y_train_balanced = self.apply_smote_and_train(X_train_processed, y_train)
        
        # Step 5: Train models on balanced data
        trained_models = self.train_models(X_train_balanced, y_train_balanced)
        
        # Step 6: Evaluate on original test set
        results, all_models, best_model, best_model_name, best_score = self.evaluate_on_original_test(
            trained_models, X_test_processed, y_test
        )
        
        # Save results
        results_df = self.save_results(results, best_model, best_model_name, best_score)
        
        # Create plots
        self.create_plots(results_df, all_models, X_test_processed, y_test)
        
        # Final summary
        print(f"\n📊 FINAL RESULTS SUMMARY")
        print("=" * 40)
        print(f"🔢 Models evaluated: {len(results)}")
        print(f"🏆 Best model: {best_model_name}")
        print(f"📈 Best ROC-AUC: {best_score:.4f}")
        
        results_df_sorted = results_df.sort_values('roc_auc', ascending=False)
        print(f"\n📋 Model Rankings:")
        for _, row in results_df_sorted.iterrows():
            print(f"   {row['model']}: ROC-AUC={row['roc_auc']:.4f}, F1={row['f1_score']:.4f}")
        
        print(f"\n⏰ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎉 Objective ML pipeline completed successfully!")
        print("\n💡 This ensures the most reliable results by:")
        print("   ✅ Using combined dataset for more data")
        print("   ✅ Immediate split for objectivity")
        print("   ✅ No data leakage (processors fit on train only)")
        print("   ✅ SMOTE applied correctly on processed data")
        print("   ✅ Evaluation on original untouched test set")


def main():
    pipeline = ChurnPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
