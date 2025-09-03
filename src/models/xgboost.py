import numpy as np
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from modules.feature_engineering import FeatureEngineeringPipeline
from models.utils import cross_val_score

seed = 42

def get_xgb_pipeline():
    params = {
        'eta': 0.04,
        'max_depth': 5,
        'subsample': 0.89,
        'colsample_bytree': 0.42,
        'min_child_weight': 0.42,
        'reg_lambda': 1.76,
        'reg_alpha': 1.99,
        'n_estimators': 1000,
        'random_state': seed,
        'tree_method': 'hist'
    }
    return make_pipeline(
        FeatureEngineeringPipeline(),
        XGBClassifier(**params)
    )
