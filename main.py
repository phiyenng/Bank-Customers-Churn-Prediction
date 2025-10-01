"""
Main Pipeline for Bank Customer Churn Prediction
===============================================

"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib

# Import from our modules
from src.modules.processing import DataLoader, OutlierHandler
from src.modules.feature_engineering import FeatureCreation, FeatureTransformation, FeatureSelection
from src.modules.imbalance_handler import ImbalanceHandler
from src.models.train_cv import run_cv_training
from src.models.evaluation import (
    get_evaluation_metrics,
    generate_evaluation_plots,
    generate_shap_plots,
)
from src.models.xgboost import XGBoostModel
from src.models.lightgbm import LightGBMModel
from src.models.catboost import CatBoostModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import optuna
from sklearn.model_selection import cross_val_score


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)



def main():
    print("Bank Customer Churn Prediction Pipeline")
    print("=" * 50)

    # ------------- Configuration -------------
    config = load_config()
    paths = config['paths']
    preprocessing = config['preprocessing']

    feature_eng_cfg = config.get('feature_engineering', {})
    feat_select_cfg = config.get('feature_selection', {})

    imbalance_cfg = config['imbalance']
    split_cfg = config['split']
    cv_cfg = config.get('cv', {"n_splits": 5})
    training_cfg = config.get('training', {"enable_baselines": True, "models": ["XGBoost", "LightGBM", "CatBoost"]})
    tuning_cfg = config.get('tuning', {"enable": False})
    
    target_column = preprocessing['target_column']

    artifacts_dir = Path(paths['artifacts_dir'])
    
    # ------------- Step 1: LOAD AND COMBINE DATA -------------
    print("\n1) LOADING AND COMBINING DATA...")
    data_loader = DataLoader(paths)
    original_df, test_df, combined_df = data_loader.get_data()
    print(f"COMBINED DATASET SHAPE: {combined_df.shape}")

    # ------------- Step 2: OUTLIER HANDLING -------------
    print("\n2) OUTLIER HANDLING...")
    if preprocessing['outlier_handling']['enable']:
        outlier_handler = OutlierHandler(
            iqr_multiplier=preprocessing['outlier_handling']['iqr_multiplier'],
            target_col=target_column
        )
        combined_df = outlier_handler.remove_outliers(combined_df, strategy=preprocessing['outlier_handling']['strategy'],
                                                            max_iter=preprocessing['outlier_handling']['max_iter'], verbose=True)
        print(f"AFTER OUTLIER REMOVAL: {combined_df.shape}")

    # ------------- Step 3: FEATURE CREATION -------------
    print("\n3) FEATURE CREATION...")
    if feature_eng_cfg.get('creation', False):
        creator = FeatureCreation()
        combined_df = creator.create(combined_df)
        print(f"AFTER FEATURE CREATION: {combined_df.shape}")

    # ------------- Step 4: SPLITTING DATA -------------
    print("\n4) SPLITTING DATA...")
    X = combined_df.drop(columns=[target_column])
    y = combined_df[target_column]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=split_cfg['test_size'],
        stratify=y,
        random_state=split_cfg['random_state']
    )
    print(f"TRAIN (FOR CV): {X_temp.shape}, TEST (HELD-OUT): {X_test.shape}")

    # Prepare train and test DataFrames with target column for further processing
    train_df = X_temp.copy()
    train_df[target_column] = y_temp.values
    test_df = X_test.copy()
    test_df[target_column] = y_test.values

    # ------------- Step 5: FEATURE TRANSFORMATION -------------
    print('\n5) FEATURE TRANSFORMATION...')
    if feature_eng_cfg.get('transformation', False):
        print('   Deferred to within each CV fold and final training to avoid leakage.')
    else:
        print('   Skipped (disabled via config).')

    # Extract features and target for CV
    X_temp = train_df.drop(columns=[target_column])
    y_temp = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    print(f"\nSHAPES BEFORE CV - TRAIN: {X_temp.shape}, TEST: {X_test.shape}")

    # ------------- Step 6: CROSS-VALIDATION SETUP (STRATIFIED K-FOLD) -------------
    print("\n6) STRATIFIED K-FOLD CV WITH FEATURE SELECTION PER FOLD...")
    skf = StratifiedKFold(n_splits=cv_cfg.get('n_splits', 5), shuffle=True, random_state=split_cfg['random_state'])

    cv_dir = artifacts_dir / 'cv'
    cv_dir.mkdir(parents=True, exist_ok=True)

    fold_num = 0
    for train_index, val_index in skf.split(X_temp, y_temp):
        fold_num += 1
        print(f"\n- PROCESSING FOLD {fold_num} ...")
        
        # Split into fold-train and fold-valid
        X_train_fold = X_temp.iloc[train_index].copy()
        y_train_fold = y_temp.iloc[train_index].copy()
        X_val_fold = X_temp.iloc[val_index].copy()
        y_val_fold = y_temp.iloc[val_index].copy()

        # Reconstruct DataFrames with target for downstream steps
        train_fold_df = X_train_fold.copy()
        train_fold_df[target_column] = y_train_fold.values

        val_fold_df = X_val_fold.copy()
        val_fold_df[target_column] = y_val_fold.values

        # ------------- FEATURE TRANSFORMATION -------------
        if feature_eng_cfg.get('transformation', False):
            print('  APPLYING FEATURE TRANSFORMATION (FIT ON FOLD-TRAIN ONLY)...')
            transformer_fold = FeatureTransformation()
            train_fold_df = transformer_fold.fit_transform(train_fold_df)
            val_fold_df = transformer_fold.transform(val_fold_df)
        
        # ------------- FEATURE SELECTION -------------
        if feat_select_cfg.get('enable', True):
            print(f"  APPLYING FEATURE SELECTION (FIT ON FOLD-TRAIN ONLY)...")
            selector = FeatureSelection(
                method=feat_select_cfg.get('method', 'correlation'),
                threshold=feat_select_cfg.get('threshold', 0.95),
                target_col=feat_select_cfg.get('target_col', target_column),
                k=feat_select_cfg.get('k'),
                percentile=feat_select_cfg.get('percentile')
            )
            train_fold_df = selector.fit_transform(train_fold_df)
            val_fold_df = selector.transform(val_fold_df)

        # Extract features and target again after transformations/selection
        X_train_fold = train_fold_df.drop(columns=[target_column])
        y_train_fold = train_fold_df[target_column]
        X_val_fold = val_fold_df.drop(columns=[target_column])
        y_val_fold = val_fold_df[target_column]

        if feat_select_cfg.get('enable', True):
            print(f"  AFTER FEATURE SELECTION - TRAIN: {X_train_fold.shape}, VAL: {X_val_fold.shape}")
        elif feature_eng_cfg.get('transformation', False):
            print(f"  AFTER FEATURE TRANSFORMATION - TRAIN: {X_train_fold.shape}, VAL: {X_val_fold.shape}")

        # ------------- HANDLE IMBALANCE PER FOLD -------------
        if imbalance_cfg.get('enable', True) and imbalance_cfg.get('method') != 'class_weight':
            imbalance_handler = ImbalanceHandler(
                method=imbalance_cfg.get('method', 'smote'),
                random_state=split_cfg['random_state']
            )
            X_train_fold, y_train_fold = imbalance_handler.fit_resample(
                X_train_fold, y_train_fold,
                sampling_strategy=imbalance_cfg.get('sampling_strategy', 'auto')
            )
        elif imbalance_cfg.get('enable', True) and imbalance_cfg.get('method') == 'class_weight':
            print("  Using class_weight method - data will not be resampled (weights computed during training)")

        fold_dir = cv_dir / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        pd.concat([X_train_fold, y_train_fold], axis=1).to_csv(str((fold_dir / 'train.csv').resolve()), index=False)
        pd.concat([X_val_fold, y_val_fold], axis=1).to_csv(str((fold_dir / 'val.csv').resolve()), index=False)
        print(f"  SAVED FOLD {fold_num} TO {fold_dir}")

    print(f"\nCV FOLD PREPARATION COMPLETED. FOLDS SAVED TO: {cv_dir}")

    # ------------- Step 7: BASELINE CV TRAINING & COMPARISON -------------
    if training_cfg.get('enable_baselines', True):
        print("\n7) BASELINE CV TRAINING & EVALUATION...")
        agg_df = run_cv_training(artifacts_dir=artifacts_dir, target_col=target_column)
        # Pick best model by ROC_AUC
        best_row = agg_df.sort_values('ROC_AUC', ascending=False).iloc[0]
        best_model_name = best_row['Model']
        print(f"BEST BASELINE MODEL BY ROC_AUC: {best_model_name}")
        
        # Train baseline on full train set and evaluate on test for fair comparison
        print(f"\n7.5) TRAINING BASELINE {best_model_name} ON FULL TRAIN SET FOR TEST EVALUATION...")
        
        wrapper_map_baseline = {
            'XGBoost': XGBoostModel,
            'LightGBM': LightGBMModel,
            'CatBoost': CatBoostModel,
        }
        BaselineWrapper = wrapper_map_baseline[best_model_name]
        
        # ------------- FEATURE SELECTION ON FULL TRAIN SET -------------
        X_baseline = X_temp.copy()
        y_baseline = y_temp.copy()
        X_test_baseline = X_test.copy()
        y_test_baseline = y_test.copy()
        
        # Reconstruct DataFrames with target for transformation/feature selection
        train_baseline_df = X_baseline.copy()
        train_baseline_df[target_column] = y_baseline.values

        test_baseline_df = X_test_baseline.copy()
        test_baseline_df[target_column] = y_test_baseline.values

        if feature_eng_cfg.get('transformation', False):
            print("     APPLYING FEATURE TRANSFORMATION ON FULL TRAIN SET...")
            transformer_baseline = FeatureTransformation()
            train_baseline_df = transformer_baseline.fit_transform(train_baseline_df)
            test_baseline_df = transformer_baseline.transform(test_baseline_df)
            print(f"     AFTER FEATURE TRANSFORMATION - TRAIN: {train_baseline_df.shape}, TEST: {test_baseline_df.shape}")

        if feat_select_cfg.get('enable', True):
            print("     APPLYING FEATURE SELECTION ON FULL TRAIN SET...")
            selector_baseline = FeatureSelection(
                method=feat_select_cfg.get('method', 'correlation'),
                threshold=feat_select_cfg.get('threshold', 0.95),
                target_col=feat_select_cfg.get('target_col', target_column),
                k=feat_select_cfg.get('k'),
                percentile=feat_select_cfg.get('percentile')
            )
            train_baseline_df = selector_baseline.fit_transform(train_baseline_df)
            test_baseline_df = selector_baseline.transform(test_baseline_df)
            print(f"     AFTER FEATURE SELECTION - TRAIN: {train_baseline_df.shape}, TEST: {test_baseline_df.shape}")

        # Extract features and target
        X_baseline = train_baseline_df.drop(columns=[target_column])
        y_baseline = train_baseline_df[target_column]
        X_test_baseline = test_baseline_df.drop(columns=[target_column])
        y_test_baseline = test_baseline_df[target_column]


        # ------------- HANDLE IMBALANCE PER FOLD -------------        
        sample_weight_baseline = None
        
        if imbalance_cfg.get('enable', True):
            if imbalance_cfg.get('method') == 'class_weight':
                imbalance_handler_bl = ImbalanceHandler(method='class_weight')
                class_weights_dict_bl = imbalance_handler_bl.get_class_weights(y_baseline, method='balanced')
                sample_weight_baseline = np.array([class_weights_dict_bl[label] for label in y_baseline])
            else:
                imbalance_handler_bl = ImbalanceHandler(
                    method=imbalance_cfg.get('method', 'smote'),
                    random_state=split_cfg['random_state']
                )
                X_baseline, y_baseline = imbalance_handler_bl.fit_resample(
                    X_baseline, y_baseline,
                    sampling_strategy=imbalance_cfg.get('sampling_strategy', 'auto')
                )
        
        baseline_model = BaselineWrapper(params={})  # Default params
        baseline_model.fit(X_baseline, y_baseline, sample_weight=sample_weight_baseline)
        
        y_pred_test_baseline = baseline_model.predict(X_test_baseline)
        y_proba_test_baseline = baseline_model.predict_proba(X_test_baseline)[:, 1]
        baseline_model_id = f"{best_model_name}_baseline"
        test_metrics_baseline = get_evaluation_metrics(
            y_test_baseline,
            y_pred_test_baseline,
            y_proba_test_baseline,
            model_id=baseline_model_id,
        )
        pd.DataFrame([test_metrics_baseline]).to_csv(artifacts_dir / 'test_metrics_baseline.csv', index=False)
        print(f"BASELINE TEST ROC_AUC: {test_metrics_baseline['ROC_AUC']:.4f}")

        generate_evaluation_plots(
            y_true=y_test_baseline,
            y_pred=y_pred_test_baseline,
            y_proba=y_proba_test_baseline,
            output_dir=artifacts_dir / 'plots',
            model_id=baseline_model_id,
            class_names=['Stayed', 'Exited'],
        )
        print(f"Saved baseline evaluation plots to: {artifacts_dir / 'plots'}")

        generate_shap_plots(
            model=baseline_model.get_native_model(),
            X=X_baseline,
            output_dir=artifacts_dir / 'plots',
            model_id=baseline_model_id,
        )
        print("Generated baseline SHAP interpretability plots (if supported).")
        
        # Save processed test set for reference
        pd.concat([X_test_baseline, y_test_baseline], axis=1).to_csv(artifacts_dir / 'test_processed.csv', index=False)
        print(f"SAVED FEATURE-SELECTED TEST SET TO: {artifacts_dir / 'test_processed.csv'}")

        # ------------- Step 8: HYPERPARAMETER TUNING WITH OPTUNA -------------
        best_params = {}
        if tuning_cfg.get('enable', False):
            print("\n8) HYPERPARAMETER TUNING WITH OPTUNA...")
            
            scoring = tuning_cfg.get('scoring', 'roc_auc')
            n_trials = tuning_cfg.get('n_trials', 50)
            param_space = tuning_cfg.get(best_model_name, {}).get('param_space', {})
            
            estimator_map = {
                'XGBoost': XGBClassifier,
                'LightGBM': LGBMClassifier,
                'CatBoost': CatBoostClassifier,
            }
            Estimator = estimator_map[best_model_name]

            def objective(trial):

                params = {}
                for name, config in param_space.items():
                    param_type = config.get('type')
                    
                    # Handle range-based parameters (low, high)
                    if 'low' in config and 'high' in config:
                        low, high = config['low'], config['high']
                        if param_type == 'int':
                            params[name] = trial.suggest_int(name, low, high)
                        elif param_type == 'float':
                            params[name] = trial.suggest_float(name, low, high, log=True)
                    
                    # Handle categorical/discrete parameters (values)
                    elif 'values' in config:
                        values = config['values']
                        if param_type == 'int':
                            params[name] = trial.suggest_int(name, min(values), max(values))
                        elif param_type == 'float':
                            params[name] = trial.suggest_categorical(name, values)
                        else:
                            params[name] = trial.suggest_categorical(name, values)
                
                # Manual cross-validation with PROPER feature selection per fold
                skf_cv = StratifiedKFold(n_splits=cv_cfg.get('n_splits', 5), shuffle=True, random_state=split_cfg['random_state'])
                scores = []
                
                for train_idx, val_idx in skf_cv.split(X_temp, y_temp):
                    X_train_cv = X_temp.iloc[train_idx].copy()
                    y_train_cv = y_temp.iloc[train_idx].copy()
                    X_val_cv = X_temp.iloc[val_idx].copy()
                    y_val_cv = y_temp.iloc[val_idx].copy()
                    
                    # Reconstruct DataFrames with target
                    train_cv_df = X_train_cv.copy()
                    train_cv_df[target_column] = y_train_cv.values

                    val_cv_df = X_val_cv.copy()
                    val_cv_df[target_column] = y_val_cv.values

                    # Apply feature transformation per fold (fit on train_cv only)
                    if feature_eng_cfg.get('transformation', False):
                        transformer_cv = FeatureTransformation()
                        train_cv_df = transformer_cv.fit_transform(train_cv_df)
                        val_cv_df = transformer_cv.transform(val_cv_df)

                    # Apply feature selection per fold (fit on train_cv only)
                    if feat_select_cfg.get('enable', True):
                        selector_cv = FeatureSelection(
                            method=feat_select_cfg.get('method', 'correlation'),
                            threshold=feat_select_cfg.get('threshold', 0.95),
                            target_col=feat_select_cfg.get('target_col', target_column),
                            k=feat_select_cfg.get('k'),
                            percentile=feat_select_cfg.get('percentile')
                        )
                        train_cv_df = selector_cv.fit_transform(train_cv_df)
                        val_cv_df = selector_cv.transform(val_cv_df)

                    X_train_cv = train_cv_df.drop(columns=[target_column])
                    y_train_cv = train_cv_df[target_column]
                    X_val_cv = val_cv_df.drop(columns=[target_column])
                    y_val_cv = val_cv_df[target_column]
                    
                    # Handle imbalance within Optuna CV fold
                    sample_weight_cv = None
                    if imbalance_cfg.get('enable', True):
                        method_cv = imbalance_cfg.get('method', 'smote')
                        if method_cv == 'class_weight':
                            imbalance_handler_cv = ImbalanceHandler(method='class_weight')
                            class_weights_dict = imbalance_handler_cv.get_class_weights(y_train_cv, method='balanced')
                            sample_weight_cv = np.array([class_weights_dict[label] for label in y_train_cv])
                        else:
                            imbalance_handler_cv = ImbalanceHandler(
                                method=method_cv,
                                random_state=split_cfg['random_state']
                            )
                            X_train_cv, y_train_cv = imbalance_handler_cv.fit_resample(
                                X_train_cv,
                                y_train_cv,
                                sampling_strategy=imbalance_cfg.get('sampling_strategy', 'auto')
                            )
                    
                    # Train model
                    model = Estimator(**params, random_state=split_cfg['random_state'])
                    model.fit(X_train_cv, y_train_cv, sample_weight=sample_weight_cv)
                    
                    # Evaluate based on scoring metric
                    if scoring == 'roc_auc':
                        from sklearn.metrics import roc_auc_score
                        y_proba = model.predict_proba(X_val_cv)[:, 1]
                        score = roc_auc_score(y_val_cv, y_proba)
                    elif scoring == 'f1':
                        from sklearn.metrics import f1_score
                        y_pred = model.predict(X_val_cv)
                        score = f1_score(y_val_cv, y_pred)
                    elif scoring == 'accuracy':
                        from sklearn.metrics import accuracy_score
                        y_pred = model.predict(X_val_cv)
                        score = accuracy_score(y_val_cv, y_pred)
                    else:
                        # Default to accuracy
                        y_pred = model.predict(X_val_cv)
                        from sklearn.metrics import accuracy_score
                        score = accuracy_score(y_val_cv, y_pred)
                    
                    scores.append(score)
                
                return np.mean(scores)

            study = optuna.create_study(direction='maximize')
            
            study.optimize(objective, n_trials=n_trials)
            
            best_params = study.best_params
            print(f"Best params for {best_model_name}: {best_params}")
            print(f"Best {scoring} score (Optuna CV): {study.best_value}")
            
            # Compare with baseline
            baseline_score = best_row['ROC_AUC']
            improvement = study.best_value - baseline_score
            print(f"\n{'='*60}")
            print(f"FAIR COMPARISON (both on CV):")
            print(f"  Baseline {best_model_name} CV ROC_AUC: {baseline_score:.4f}")
            print(f"  Tuned {best_model_name} CV ROC_AUC:    {study.best_value:.4f}")
            print(f"  Improvement:                          {improvement:+.4f} ({improvement/baseline_score*100:+.2f}%)")
            print(f"{'='*60}\n")

            # Step 9: Final training on full train set and evaluation on held-out test
            print("\n9) Final training on full train set and evaluation on held-out test...")
            
            # Apply transformations / feature selection on FULL train set
            train_final_df = X_temp.copy()
            train_final_df[target_column] = y_temp.values

            test_final_df = X_test.copy()
            test_final_df[target_column] = y_test.values

            if feature_eng_cfg.get('transformation', False):
                print("   APPLYING FEATURE TRANSFORMATION ON FULL TRAIN SET...")
                transformer_final = FeatureTransformation()
                train_final_df = transformer_final.fit_transform(train_final_df)
                test_final_df = transformer_final.transform(test_final_df)
                print(f"   AFTER FEATURE TRANSFORMATION - TRAIN: {train_final_df.shape}, TEST: {test_final_df.shape}")

            if feat_select_cfg.get('enable', True):
                print("   APPLYING FEATURE SELECTION ON FULL TRAIN SET...")
                selector_final = FeatureSelection(
                    method=feat_select_cfg.get('method', 'correlation'),
                    threshold=feat_select_cfg.get('threshold', 0.95),
                    target_col=feat_select_cfg.get('target_col', target_column),
                    k=feat_select_cfg.get('k'),
                    percentile=feat_select_cfg.get('percentile')
                )
                train_final_df = selector_final.fit_transform(train_final_df)
                test_final_df = selector_final.transform(test_final_df)
                print(f"   AFTER FEATURE SELECTION - TRAIN: {train_final_df.shape}, TEST: {test_final_df.shape}")

            # Extract features and target
            X_final = train_final_df.drop(columns=[target_column])
            y_final = train_final_df[target_column]
            X_test_final = test_final_df.drop(columns=[target_column])
            y_test_final = test_final_df[target_column]
            
            sample_weight_final = None
            
            if imbalance_cfg.get('enable', True):
                imbalance_handler = ImbalanceHandler(
                    method=imbalance_cfg.get('method', 'smote'),
                    random_state=split_cfg['random_state']
                )
                
                if imbalance_cfg.get('method') == 'class_weight':
                    # Compute sample weights instead of resampling
                    class_weights_dict = imbalance_handler.get_class_weights(y_final, method='balanced')
                    sample_weight_final = np.array([class_weights_dict[label] for label in y_final])
                else:
                    # Resample data for other methods
                    X_final, y_final = imbalance_handler.fit_resample(
                        X_final, y_final,
                        sampling_strategy=imbalance_cfg.get('sampling_strategy', 'auto')
                    )

            wrapper_map = {
                'XGBoost': XGBoostModel,
                'LightGBM': LightGBMModel,
                'CatBoost': CatBoostModel,
            }
            BestWrapper = wrapper_map[best_model_name]
            
            final_model = BestWrapper(params=best_params) 
            final_model.fit(X_final, y_final, sample_weight=sample_weight_final)

            y_pred_test = final_model.predict(X_test_final)
            y_proba_test = final_model.predict_proba(X_test_final)[:, 1]
            tuned_model_id = f"{best_model_name}_tuned_optuna"
            test_metrics = get_evaluation_metrics(y_test_final, y_pred_test, y_proba_test, model_id=tuned_model_id)
            pd.DataFrame([test_metrics]).to_csv(artifacts_dir / 'test_metrics_optuna.csv', index=False)

            generate_evaluation_plots(
                y_true=y_test_final,
                y_pred=y_pred_test,
                y_proba=y_proba_test,
                output_dir=artifacts_dir / 'plots',
                model_id=tuned_model_id,
                class_names=['Stayed', 'Exited'],
            )
            print(f"Saved tuned model evaluation plots to: {artifacts_dir / 'plots'}")

            generate_shap_plots(
                model=final_model.get_native_model(),
                X=X_final,
                output_dir=artifacts_dir / 'plots',
                model_id=tuned_model_id,
            )
            print("Generated tuned model SHAP interpretability plots (if supported).")
            
            print("\n" + "="*60)
            print("HELD-OUT TEST SET PERFORMANCE (Final Reality Check):")
            print("="*60)
            print("Note: Test metrics are typically LOWER than CV metrics")
            print("      because test set is completely unseen data.\n")
            print(pd.DataFrame([test_metrics]))
            print("="*60)

            # Save tuned best model
            joblib.dump(final_model.model, artifacts_dir / 'best_model_tuned_optuna.joblib')
            
            # Final comprehensive comparison
            print("\n" + "="*70)
            print("COMPREHENSIVE PERFORMANCE COMPARISON")
            print("="*70)
            print("\n1. CROSS-VALIDATION METRICS (used for model selection):")
            print(f"   Baseline {best_model_name} (default params): ROC_AUC = {baseline_score:.4f}")
            print(f"   Tuned {best_model_name} (Optuna):            ROC_AUC = {study.best_value:.4f}")
            print(f"   CV Improvement:                              {improvement:+.4f} ({improvement/baseline_score*100:+.2f}%)")
            
            print("\n2. HELD-OUT TEST SET METRICS (final reality check):")
            print(f"   Baseline {best_model_name} on test:          ROC_AUC = {test_metrics_baseline['ROC_AUC']:.4f}")
            print(f"   Tuned {best_model_name} on test:             ROC_AUC = {test_metrics['ROC_AUC']:.4f}")
            test_improvement = test_metrics['ROC_AUC'] - test_metrics_baseline['ROC_AUC']
            print(f"   Test Improvement:                            {test_improvement:+.4f} ({test_improvement/test_metrics_baseline['ROC_AUC']*100:+.2f}%)")
            
            print("\n3. INTERPRETATION:")
            if test_improvement > 0:
                print("   ✓ Hyperparameter tuning IMPROVED generalization!")
                print("     The tuned model performs better on unseen data.")
            elif abs(test_improvement) < 0.005:
                print("   ~ Tuning had MINIMAL impact on test performance.")
                print("     Default parameters were already good.")
            else:
                print("   ✗ Warning: Tuned model performs WORSE on test set.")
                print("     Possible overfitting to CV folds.")
                print("     Consider: 1) Broader search space, 2) More CV folds, 3) Regularization")
            
            print("\n   Note: CV scores > Test scores is NORMAL and EXPECTED.")
            print("         Always trust test set performance for final model selection.")
            print("="*70 + "\n")

    print("Pipeline completed!")

if __name__ == "__main__":
    main()