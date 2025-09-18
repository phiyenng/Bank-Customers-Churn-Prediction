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
from src.models.evaluation import get_evaluation_metrics
from src.models.xgboost import XGBoostModel
from src.models.lightgbm import LightGBMModel
from src.models.catboost import CatBoostModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def main():
    print("Bank Customer Churn Prediction Pipeline")
    print("=" * 50)

    # Load configuration
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
    
    # Step 1: Load and combine data
    print("\n1) Loading and combining data...")
    data_loader = DataLoader(paths)
    original_df, test_df, combined_df = data_loader.get_data()
    print(f"Combined dataset shape: {combined_df.shape}")

    # Step 2: Outlier handling
    print("\n2) Outlier handling (optional)...")
    if preprocessing['outlier_handling']['enable']:
        print("Handling outliers using IQR...")
        outlier_handler = OutlierHandler(
            iqr_multiplier=preprocessing['outlier_handling']['iqr_multiplier'],
            target_col=target_column
        )
        combined_df = outlier_handler.remove_outliers(
            combined_df,
            strategy=preprocessing['outlier_handling']['strategy'],
            max_iter=preprocessing['outlier_handling']['max_iter'],
            verbose=True
        )
        print(f"After outlier removal: {combined_df.shape}")

    # Step 3: Feature Creation
    print("\n3) Feature Creation...")
    if feature_eng_cfg.get('creation', False):
        creator = FeatureCreation()
        combined_df = creator.create(combined_df)
        print(f"After feature creation: {combined_df.shape}")

    # Step 4: Feature Transformation
    print('\n4) Feature Transformation...')
    if feature_eng_cfg.get('transformation', False):
        transformer = FeatureTransformation()
        combined_df = transformer.fit_transform(combined_df)
        print(f"After feature transformation: {combined_df.shape}")

    print('\4.5) Post-Creation Cleanup...')
    cols_to_drop = ['Gender', 'Age', 'NumOfProducts', 'Geography_Germany', 'Geography_Spain', 'Total_Products_Used']
    existing_cols_to_drop = [col for col in cols_to_drop if col in combined_df.columns]
    
    if existing_cols_to_drop:
        combined_df = combined_df.drop(columns=existing_cols_to_drop)
        print(f"Dropped original columns: {existing_cols_to_drop}")
        print(f"After cleanup: {combined_df.shape}")

    # Step 5: Feature Selection
    print("\n5) Feature selection (optional)...")
    if feat_select_cfg.get('enable', True):
        selector = FeatureSelection(
            method=feat_select_cfg.get('method', 'correlation'),
            threshold=feat_select_cfg.get('threshold', 0.95),
            target_col=feat_select_cfg.get('target_col', target_column),
            k=feat_select_cfg.get('k'),
            percentile=feat_select_cfg.get('percentile')
        )
        combined_df = selector.fit_transform(combined_df)
        print(f"After feature selection: {combined_df.shape}")

    # Save processed combined dataset before split
    print("\nSaving combined processed dataset...")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    processed_path = artifacts_dir / "combined_processed.csv"
    combined_df.to_csv(processed_path, index=False)
    print(f"Combined processed dataset saved to: {processed_path}")


    # Step 6: Train/Validation/Test Split
    # Prepare features and target
    X = combined_df.drop(columns=[target_column])
    y = combined_df[target_column]
  
    print("\n6) Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=split_cfg['test_size'],
        stratify=y,
        random_state=split_cfg['random_state']
    )
    print(f"CV base: {X_temp.shape}, Test (held-out): {X_test.shape}")

    # Step 7: Cross-Validation setup (StratifiedKFold)
    print("\n7) Setting up Stratified K-Fold CV...")
    skf = StratifiedKFold(n_splits=cv_cfg.get('n_splits', 5), shuffle=True, random_state=split_cfg['random_state'])

    artifacts_dir = Path(paths['artifacts_dir'])
    cv_dir = artifacts_dir / 'cv'
    cv_dir.mkdir(parents=True, exist_ok=True)

    fold_num = 0
    for train_index, val_index in skf.split(X_temp, y_temp):
        fold_num += 1
        print(f"- Preparing fold {fold_num} ...")
        X_train_fold = X_temp.iloc[train_index]
        y_train_fold = y_temp.iloc[train_index]
        X_val_fold = X_temp.iloc[val_index]
        y_val_fold = y_temp.iloc[val_index]

        # Handle imbalance per fold
        if imbalance_cfg.get('enable', True):
            imbalance_handler = ImbalanceHandler(
                method=imbalance_cfg.get('method', 'smote'),
                random_state=split_cfg['random_state']
            )
            X_train_fold, y_train_fold = imbalance_handler.fit_resample(
                X_train_fold, y_train_fold,
                sampling_strategy=imbalance_cfg.get('sampling_strategy', 'auto')
            )

        fold_dir = cv_dir / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        pd.concat([X_train_fold, y_train_fold], axis=1).to_csv(fold_dir / 'train.csv', index=False)
        pd.concat([X_val_fold, y_val_fold], axis=1).to_csv(fold_dir / 'val.csv', index=False)
        print(f"  Saved fold {fold_num} to {fold_dir}")

    # Save processed data
    print("\nSaving processed data...")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([X_test, y_test], axis=1).to_csv(artifacts_dir / 'test_processed.csv', index=False)

    print(f"Processed data saved to: {artifacts_dir}")

    # Step 8: Baseline CV training & comparison (simple)
    if training_cfg.get('enable_baselines', True):
        print("\n8) Baseline CV training & evaluation...")
        agg_df = run_cv_training(artifacts_dir=artifacts_dir, target_col=target_column)
        # Pick best model by ROC_AUC
        best_row = agg_df.sort_values('ROC_AUC', ascending=False).iloc[0]
        best_model_name = best_row['Model']
        print(f"Best baseline model by ROC_AUC: {best_model_name}")

        # Step 9: Hyperparameter tuning (RandomizedSearchCV)
        if tuning_cfg.get('enable', False):
            print("\n9) Hyperparameter tuning with RandomizedSearchCV...")
            scoring = tuning_cfg.get('scoring', 'roc_auc')
            n_iter = tuning_cfg.get('n_iter', 20)
            param_spaces = tuning_cfg.get(best_model_name, {}).get('param_distributions', {})
            skf = StratifiedKFold(n_splits=cv_cfg.get('n_splits', 5), shuffle=True, random_state=split_cfg['random_state'])

            estimator_map = {
                'XGBoost': XGBClassifier,
                'LightGBM': LGBMClassifier,
                'CatBoost': CatBoostClassifier,
            }
            Estimator = estimator_map[best_model_name]
            base_estimator = Estimator()

            tuner = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=param_spaces,
                n_iter=n_iter,
                scoring=scoring,
                cv=skf,
                random_state=split_cfg['random_state'],
                n_jobs=-1,
                verbose=1,
            )
            tuner.fit(X_temp, y_temp)
            best_params = tuner.best_params_
            print(f"Best params for {best_model_name}: {best_params}")

            # Step 10: Final training on full CV base and evaluation on held-out test
            print("\n10) Final training on full CV base and evaluation on held-out test...")
            X_final, y_final = X_temp, y_temp
            if imbalance_cfg.get('enable', True):
                imbalance_handler = ImbalanceHandler(
                    method=imbalance_cfg.get('method', 'smote'),
                    random_state=split_cfg['random_state']
                )
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
            final_model.fit(X_final, y_final)

            y_pred_test = final_model.predict(X_test)
            y_proba_test = final_model.predict_proba(X_test)[:, 1]
            test_metrics = get_evaluation_metrics(y_test, y_pred_test, y_proba_test, model_id=f"{best_model_name}_tuned")
            pd.DataFrame([test_metrics]).to_csv(artifacts_dir / 'test_metrics.csv', index=False)
            print("\nTest metrics:")
            print(pd.DataFrame([test_metrics]))

            # Save tuned best model
            joblib.dump(final_model.model, artifacts_dir / 'best_model_tuned.joblib')

    print("Pipeline completed!")

if __name__ == "__main__":
    main()