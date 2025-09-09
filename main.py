import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_curve, classification_report, roc_auc_score, average_precision_score, accuracy_score

from src.modules.processing import DataLoader, OutlierHandler, Transformation, ImbalanceHandler
from src.models.xgboost import XGBoostModel
from src.models.lightgbm import LightGBMModel
from src.models.catboost import CatBoostModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(path: str = 'config.yaml') -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def threshold_from_pr(y_true: np.ndarray, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * (precision * recall) / np.clip(precision + recall, a_min=1e-12, a_max=None)
    return float(thresholds[np.nanargmax(f1)]) if thresholds.size > 0 else 0.5


def run_pipeline() -> None:
    cfg = load_config()
    paths = cfg['paths']
    pp = cfg['preprocessing']
    split_cfg = cfg['split']
    imb = cfg['imbalance']
    models_cfg = cfg['models']

    target = pp.get('target_column', 'Exited')

    # Step 0: Load & Combine Data
    dl = DataLoader()
    combined_df = dl.combine_datasets()
    if combined_df is None or combined_df.empty:
        raise RuntimeError('No combined data. Please verify DataLoader.combine_datasets().')

    if dl.competition_test_df is None:
        _, _ = dl.load_competition_datasets()
    test_df = dl.competition_test_df
    if test_df is None or test_df.empty:
        raise RuntimeError('No test data found from DataLoader.load_competition_datasets().')

    X_combine = combined_df.drop(columns=[target])
    y_combine = combined_df[target]

    if target in test_df.columns:
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target]
    else:
        X_test = test_df.copy()
        y_test = pd.Series([None] * len(test_df), name=target)

    # Outlier handling
    oh = OutlierHandler(iqr_multiplier=pp.get('iqr_multiplier', 1.5), target_col=target)
    combined = pd.concat([X_combine, y_combine], axis=1)
    combined = oh.remove_outliers(combined, strategy=pp.get('strategy', 'any'), max_iter=pp.get('max_iter', 10), verbose=pp.get('verbose', True))
    X_combine = combined.drop(columns=[target])
    y_combine = combined[target]

    # Step 1: Stratified Split Train/Validation
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_combine, y_combine, test_size=split_cfg['test_size'], stratify=y_combine, random_state=split_cfg['random_state']
    )

    # Scale/Transform
    tr = Transformation(handle_categorical=True)
    tr.fit(X_train_raw)
    X_train = tr.transform(X_train_raw)
    X_val = tr.transform(X_val_raw)
    X_test_scaled = tr.transform(X_test)

    # Step 2 & 3: Loop through multiple resampling methods & train models
    resampling_methods = ['smote', 'adasyn', 'smote_tomek', 'smote_enn']
    trained = {}  # store models per resampling method
    validation_reports = []  # store metrics

    for method in resampling_methods:
        print(f"\n=== Resampling method: {method} ===")
        
        # Initialize imbalance handler
        ih = ImbalanceHandler(method=method, random_state=split_cfg['random_state'])
        
        # Fit and resample (train set only)
        X_train_res, y_train_res = ih.fit_resample(X_train, y_train, sampling_strategy=imb.get('sampling_strategy', 'auto'))
        
        for name, spec in models_cfg.items():
            if not spec.get('train', False):
                continue
            
            # Initialize model
            if name == 'xgboost':
                model = XGBoostModel(params=spec.get('params', {}))
            elif name == 'lightgbm':
                model = LightGBMModel(params=spec.get('params', {}))
            elif name == 'catboost':
                model = CatBoostModel(params=spec.get('params', {}))
            else:
                continue
            
            # Train model
            model.fit(X_train_res.values, y_train_res.values, eval_set=(X_val.values, y_val.values))
            
            # Predict on validation
            proba_val = model.predict_proba(X_val.values)[:, 1]
            thr = threshold_from_pr(y_val.values, proba_val)
            y_pred_val = (proba_val >= thr).astype(int)
            
            # Save metrics
            report_row = {
                'Resampling': method,
                'Model': name,
                'Threshold': thr,
                'Accuracy': (y_val == y_pred_val).mean(),
                'F1': classification_report(y_val, y_pred_val, output_dict=True)['weighted avg']['f1-score'],
                'Precision': classification_report(y_val, y_pred_val, output_dict=True)['weighted avg']['precision'],
                'Recall': classification_report(y_val, y_pred_val, output_dict=True)['weighted avg']['recall'],
                'ROC_AUC': roc_auc_score(y_val, proba_val),
                'PR_AUC': average_precision_score(y_val, proba_val)
            }
            validation_reports.append(report_row)
            
            # Save trained model
            trained[f"{method}_{name}"] = model

    # Convert to DataFrame for easy comparison
    val_report_df = pd.DataFrame(validation_reports)
    print("\nValidation Report (all resampling methods & models):\n", val_report_df)


    # Step 4-5: Threshold tuning & Validation evaluation
    report_rows = []
    thresholds = {}
    for name, model in trained.items():
        proba_val = model.predict_proba(X_val.values)[:, 1]
        thr = threshold_from_pr(y_val.values, proba_val)
        thresholds[name] = thr
        y_pred_val = (proba_val >= thr).astype(int)

        report = classification_report(y_val, y_pred_val, output_dict=True)
        row = {
            'Model': name,
            'Accuracy': accuracy_score(y_val, y_pred_val),
            'F1': report['weighted avg']['f1-score'],
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'ROC_AUC': roc_auc_score(y_val, proba_val),
            'PR_AUC': average_precision_score(y_val, proba_val),
            'Threshold': thr
        }
        report_rows.append(row)

    val_report = pd.DataFrame(report_rows).set_index('Model')
    print("Validation Summary:\n", val_report)

    # Step 6: Evaluate best on Test
    best_name = val_report['F1'].idxmax()
    best_model = trained[best_name]
    best_thr = thresholds[best_name]

    proba_test = best_model.predict_proba(X_test_scaled.values)[:, 1]
    y_pred_test = (proba_test >= best_thr).astype(int)
    print(f"\nBest Model: {best_name} @ threshold={best_thr:.3f}")

    if y_test is not None and y_test.notnull().all() and set(pd.Series(y_test).unique()) <= {0, 1}:
        print(classification_report(y_test, y_pred_test))
        print("Accuracy:", accuracy_score(y_test, y_pred_test))
        print("ROC-AUC:", roc_auc_score(y_test, proba_test))
        print("PR-AUC:", average_precision_score(y_test, proba_test))
    else:
        out_path = Path(cfg['paths'].get('artifacts_dir', 'saved_models')) / 'test_predictions.csv'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'proba': proba_test}).to_csv(out_path, index=False)
        print(f"Saved test probabilities to {out_path}")


def main() -> None:
    run_pipeline()


if __name__ == '__main__':
    main()
