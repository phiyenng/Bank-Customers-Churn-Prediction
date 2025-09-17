"""
Cross-validated baseline training for Gradient Boosting models.

This script:
- Loads CV folds from saved_models/cv/fold_{i}/train.csv and val.csv
- Trains XGBoost, LightGBM, and CatBoost wrappers per fold
- Evaluates ROC-AUC, PR-AUC, Precision, Recall, F1 (weighted)
- Prints and saves averaged metrics to saved_models/cv/metrics_summary.csv

Usage: python -m src.models.train_cv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from .xgboost import XGBoostModel
from .lightgbm import LightGBMModel
from .catboost import CatBoostModel
from .evaluation import get_evaluation_metrics


def _load_yaml(path: Path) -> Dict:
    """Load YAML file if exists, otherwise return empty dict."""
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_fold_data(fold_dir: Path, target_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train/val CSVs for a given fold and split X/y."""
    train_df = pd.read_csv(fold_dir / "train.csv")
    val_df = pd.read_csv(fold_dir / "val.csv")
    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_val, y_val = val_df.drop(columns=[target_col]), val_df[target_col]
    return X_train, y_train, X_val, y_val


def _evaluate_on_fold(model_id: str, model_obj, X_train, y_train, X_val, y_val) -> Dict[str, float]:
    """Fit model and compute evaluation metrics dictionary for one fold."""
    model_obj.fit(X_train, y_train)
    y_pred = model_obj.predict(X_val)
    y_proba = model_obj.predict_proba(X_val)[:, 1]
    return get_evaluation_metrics(y_val, y_pred, y_proba, model_id=model_id)


def run_cv_training(artifacts_dir: Path = Path("saved_models"), target_col: str = "Exited") -> pd.DataFrame:
    """
    Run cross-validated training for baseline models and return a metrics summary DataFrame.
    """
    cv_dir = artifacts_dir / "cv"
    fold_dirs = sorted([p for p in cv_dir.glob("fold_*") if p.is_dir()])
    if not fold_dirs:
        raise FileNotFoundError(f"No folds found under {cv_dir}. Run the pipeline to generate CV folds first.")

    # Load optional per-model configs
    configs_root = Path("configs")
    xgb_params = _load_yaml(configs_root / "xgboost_config.yaml").get("params", {})
    lgb_params = _load_yaml(configs_root / "lightgbm_config.yaml").get("params", {})
    ctb_params = _load_yaml(configs_root / "catboost_config.yaml").get("params", {})

    # Initialize model constructors (keep it DRY)
    model_specs = [
        ("XGBoost", XGBoostModel, xgb_params),
        ("LightGBM", LightGBMModel, lgb_params),
        ("CatBoost", CatBoostModel, ctb_params),
    ]

    all_results: List[Dict[str, float]] = []

    for fold_dir in fold_dirs:
        X_train, y_train, X_val, y_val = _load_fold_data(fold_dir, target_col)

        for model_id, cls, params in model_specs:
            model = cls(params=params)
            metrics = _evaluate_on_fold(model_id, model, X_train, y_train, X_val, y_val)
            metrics["Fold"] = fold_dir.name
            all_results.append(metrics)

    results_df = pd.DataFrame(all_results)

    # Aggregate metrics per model (exclude non-numeric)
    agg_df = (
        results_df.groupby("Model")[["F1", "Precision", "Recall", "ROC_AUC", "PR_AUC"]]
        .mean()
        .reset_index()
        .sort_values("ROC_AUC", ascending=False)
    )

    # Save outputs
    out_file = cv_dir / "metrics_summary.csv"
    agg_df.to_csv(out_file, index=False)
    print("\nCV Results (per fold):\n", results_df)
    print("\nAveraged Metrics by Model:\n", agg_df)
    print(f"\nSaved summary to: {out_file}")

    return agg_df


if __name__ == "__main__":
    run_cv_training()


