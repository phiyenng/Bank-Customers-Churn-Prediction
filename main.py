"""
Main Pipeline for Bank Customer Churn Prediction
===============================================

Simple and clean pipeline using the updated modules.
"""

import logging
import random
import gc
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score

# Import from our modules
from src.modules.processing import DataLoader, OutlierHandler
from src.modules.feature_engineering import (
    Transformation, 
    apply_all_feature_engineering,
    add_age_category,
    add_credit_score_range,
    add_balance_salary_ratio,
    add_geo_gender,
    add_total_products_used,
    add_tp_gender,
    select_features,
    reduce_features
)
from src.modules.imbalance_handler import ImbalanceHandler
from src.models.xgboost import XGBoostModel
from src.models.lightgbm import LightGBMModel
from src.models.catboost import CatBoostModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """Load configuration from YAML file."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def set_global_seed(seed: int) -> None:
    """Set seeds for all relevant libraries to ensure reproducibility per run."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        if hasattr(torch, 'manual_seed'):
            torch.manual_seed(seed)
            if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'manual_seed_all'):
                torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch is optional; ignore if not installed
        pass

def create_model(model_name: str, model_config: dict):
    """Factory to create a fresh, unfitted model instance from config every time."""
    params = model_config.get('params', {})
    if model_name == 'xgboost':
        return XGBoostModel(params=params)
    if model_name == 'lightgbm':
        return LightGBMModel(params=params)
    if model_name == 'catboost':
        return CatBoostModel(params=params)
    raise ValueError(f"Unknown model '{model_name}' in configuration")

def get_best_threshold(y_true, y_proba):
    """Find best threshold using precision-recall curve."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5

def apply_feature_engineering(df, config):
    """Apply feature engineering based on configuration."""
    if not config.get('enable', True):
        return df
    
    print("Applying feature engineering...")
    
    if config.get('apply_all_features', True):
        # Apply all feature engineering functions
        df = apply_all_feature_engineering(df)
        print("Applied all feature engineering functions")
    else:
        # Apply specific features
        custom_features = config.get('custom_features', [])
        feature_map = {
            'age_category': add_age_category,
            'credit_score_range': add_credit_score_range,
            'balance_salary_ratio': add_balance_salary_ratio,
            'geo_gender': add_geo_gender,
            'total_products_used': add_total_products_used,
            'tp_gender': add_tp_gender
        }
        
        for feature in custom_features:
            if feature in feature_map:
                df = feature_map[feature](df)
                print(f"Applied {feature}")
    
    return df

def apply_feature_selection(df, config, target_column, fitted_selectors=None):
    """Apply feature selection based on configuration."""
    if not config.get('enable', False):
        return df, fitted_selectors
    
    print("Applying feature selection...")
    
    method = config.get('method', 'correlation')
    threshold = config.get('threshold', 0.95)
    k = config.get('k')
    percentile = config.get('percentile')
    
    # For now, we'll use the simple approach - fit and transform in one step
    # In production, you'd want to save the fitted selectors and reuse them
    df = select_features(
        df, 
        method=method, 
        threshold=threshold,
        target_col=target_column,
        k=k,
        percentile=percentile
    )
    
    print(f"Feature selection completed using {method}")
    return df, fitted_selectors

def apply_feature_reduction(df, config, target_column, fitted_reducers=None):
    """Apply feature reduction based on configuration."""
    if not config.get('enable', False):
        return df, fitted_reducers
    
    print("Applying feature reduction...")
    
    method = config.get('method', 'pca')
    n_components = config.get('n_components', 10)
    random_state = config.get('random_state', 42)
    
    # For now, we'll use the simple approach - fit and transform in one step
    # In production, you'd want to save the fitted reducers and reuse them
    df = reduce_features(
        df,
        method=method,
        n_components=n_components,
        target_col=target_column,
        random_state=random_state
    )
    
    print(f"Feature reduction completed using {method}")
    return df, fitted_reducers

def main():
    """Main pipeline function."""
    print("Starting Bank Customer Churn Prediction Pipeline")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    paths = config['paths']
    preprocessing = config['preprocessing']
    feature_eng_config = config.get('feature_engineering', {})
    feature_selection_config = config.get('feature_selection', {})
    feature_reduction_config = config.get('feature_reduction', {})
    split_config = config['split']
    imbalance_config = config['imbalance']
    models_config = config['models']
    
    target_column = preprocessing['target_column']

    # Ensure reproducibility per run
    set_global_seed(split_config.get('random_state', 42))
    
    # Step 1: Load and combine data
    print("\n1. Loading and combining data...")
    data_loader = DataLoader(paths)
    original_df, test_df, combined_df = data_loader.get_data()
    
    print(f"Combined dataset shape: {combined_df.shape}")
    
    # Step 2: Apply feature engineering
    print("\n2. Applying feature engineering...")
    combined_df = apply_feature_engineering(combined_df, feature_eng_config)
    print(f"After feature engineering: {combined_df.shape}")
    
    # Step 2.1: Apply feature selection
    print("\n2.1. Applying feature selection...")
    combined_df, fitted_selectors = apply_feature_selection(combined_df, feature_selection_config, target_column)
    print(f"After feature selection: {combined_df.shape}")
    
    # Step 2.2: Apply feature reduction (optional)
    print("\n2.2. Applying feature reduction...")
    combined_df, fitted_reducers = apply_feature_reduction(combined_df, feature_reduction_config, target_column)
    print(f"After feature reduction: {combined_df.shape}")
    
    # Step 3: Handle outliers
    print("\n3. Handling outliers...")
    outlier_handler = OutlierHandler(
        iqr_multiplier=preprocessing['iqr_multiplier'],
        target_col=target_column
    )
    combined_df = outlier_handler.remove_outliers(
        combined_df,
        strategy=preprocessing['strategy'],
        max_iter=preprocessing['max_iter'],
        verbose=preprocessing['verbose']
    )
    print(f"After outlier removal: {combined_df.shape}")
    
    # Step 4: Prepare features and target
    X = combined_df.drop(columns=[target_column])
    y = combined_df[target_column]
    
    # Step 5: Split data
    print("\n4. Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=split_config['test_size'],
        stratify=y,
        random_state=split_config['random_state']
    )
    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")
    
    # Step 6: Transform features
    print("\n5. Transforming features...")
    transformer = Transformation(handle_categorical=True)
    transformer.fit(X_train)
    X_train_transformed = transformer.transform(X_train)
    X_val_transformed = transformer.transform(X_val)
    print(f"Transformed features: {X_train_transformed.shape[1]} columns")
    
    # Step 7: Train models with different resampling methods
    print("\n6. Training models...")
    results = []
    trained_models = {}
    
    resampling_methods = imbalance_config['methods']
    
    for method in resampling_methods:
        print(f"\n--- Using {method} resampling ---")
        
        # Apply resampling
        imbalance_handler = ImbalanceHandler(method=method, random_state=split_config['random_state'])
        X_train_resampled, y_train_resampled = imbalance_handler.fit_resample(
            X_train_transformed, y_train,
            sampling_strategy=imbalance_config['sampling_strategy']
        )
        print(f"Resampled train set: {X_train_resampled.shape}")
        
        # Train each model
        for model_name, model_config in models_config.items():
            if not model_config.get('train', True):
                continue
                
            print(f"Training {model_name}...")
            
            # Initialize a fresh model from config (no reuse of fitted instances)
            model = create_model(model_name, model_config)
            
            # Train model
            model.fit(
                X_train_resampled.values, 
                y_train_resampled.values,
                eval_set=(X_val_transformed.values, y_val.values)
            )
            
            # Make predictions
            y_proba = model.predict_proba(X_val_transformed.values)[:, 1]
            threshold = get_best_threshold(y_val.values, y_proba)
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, y_proba)
            pr_auc = average_precision_score(y_val, y_proba)
            
            # Get F1 score from classification report
            report = classification_report(y_val, y_pred, output_dict=True)
            f1_score = report['weighted avg']['f1-score']
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            
            # Store results
            result = {
                'Resampling': method,
                'Model': model_name,
                'Accuracy': accuracy,
                'F1': f1_score,
                'Precision': precision,
                'Recall': recall,
                'ROC_AUC': roc_auc,
                'PR_AUC': pr_auc,
                'Threshold': threshold
            }
            results.append(result)
            
            # Store trained model
            model_key = f"{method}_{model_name}"
            trained_models[model_key] = model
            
            print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    # Step 8: Display results
    print("\n7. Results Summary:")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Find best model
    best_result = results_df.loc[results_df['F1'].idxmax()]
    best_model_key = f"{best_result['Resampling']}_{best_result['Model']}"
    best_model = trained_models[best_model_key]
    
    print(f"\nBest Model: {best_model_key}")
    print(f"Best F1 Score: {best_result['F1']:.4f}")
    print(f"Best Accuracy: {best_result['Accuracy']:.4f}")
    
    # Step 9: Make predictions on test set
    print("\n8. Making predictions on test set...")
    if target_column in test_df.columns:
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        
        # Apply same feature engineering pipeline as training data
        print("Applying feature engineering to test set...")
        X_test = apply_feature_engineering(X_test, feature_eng_config)
        # Apply selection on test only for unsupervised methods
        sel_method = feature_selection_config.get('method', 'correlation') if feature_selection_config else 'correlation'
        if feature_selection_config.get('enable', False) and sel_method in ['correlation', 'variance']:
            X_test, _ = apply_feature_selection(X_test, feature_selection_config, target_column)
        X_test, _ = apply_feature_reduction(X_test, feature_reduction_config, target_column)
        
        # Apply same transformations
        X_test_transformed = transformer.transform(X_test)
        # Align columns with training features
        X_test_transformed = X_test_transformed.reindex(columns=X_train_transformed.columns, fill_value=0)
        
        print(f"Test set shape after processing: {X_test_transformed.shape}")
        print(f"Train set shape: {X_train_transformed.shape}")
        
        # Make predictions
        y_test_proba = best_model.predict_proba(X_test_transformed.values)[:, 1]
        y_test_pred = (y_test_proba >= best_result['Threshold']).astype(int)
        
        # Calculate test metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
        test_pr_auc = average_precision_score(y_test, y_test_proba)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test ROC-AUC: {test_roc_auc:.4f}")
        print(f"Test PR-AUC: {test_pr_auc:.4f}")
        
        print("\nTest Classification Report:")
        print(classification_report(y_test, y_test_pred))
    else:
        # No target in test set, save predictions
        X_test = test_df.copy()
        
        # Apply same feature engineering pipeline as training data
        print("Applying feature engineering to test set...")
        X_test = apply_feature_engineering(X_test, feature_eng_config)
        sel_method = feature_selection_config.get('method', 'correlation') if feature_selection_config else 'correlation'
        if feature_selection_config.get('enable', False) and sel_method in ['correlation', 'variance']:
            X_test, _ = apply_feature_selection(X_test, feature_selection_config, target_column)
        X_test, _ = apply_feature_reduction(X_test, feature_reduction_config, target_column)
        
        # Apply same transformations
        X_test_transformed = transformer.transform(X_test)
        # Align columns with training features
        X_test_transformed = X_test_transformed.reindex(columns=X_train_transformed.columns, fill_value=0)
        
        print(f"Test set shape after processing: {X_test_transformed.shape}")
        
        y_test_proba = best_model.predict_proba(X_test_transformed.values)[:, 1]
        
        # Save predictions
        output_path = Path(paths['artifacts_dir']) / 'test_predictions.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        predictions_df = pd.DataFrame({
            'Exited_Probability': y_test_proba,
            'Exited_Prediction': (y_test_proba >= best_result['Threshold']).astype(int)
        })
        predictions_df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
    
    print("\nPipeline completed successfully!")

    # --- Cleanup to guarantee clean state across runs ---
    try:
        del original_df
    except Exception:
        pass
    try:
        del test_df, combined_df
    except Exception:
        pass
    try:
        del X, y, X_train, X_val, y_train, y_val
    except Exception:
        pass
    try:
        del transformer, X_train_transformed, X_val_transformed
    except Exception:
        pass
    try:
        del X_test, results_df, best_model, best_result
    except Exception:
        pass
    try:
        # Explicitly drop references to trained models and results
        trained_models.clear()
        del trained_models
    except Exception:
        pass
    try:
        del results
    except Exception:
        pass
    # Trigger garbage collection for native memory used by boosters/estimators
    gc.collect()

if __name__ == "__main__":
    main()