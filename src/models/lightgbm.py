from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from modules.feature_engineering import FeatureEngineeringPipeline
from models.utils import cross_val_score

seed = 42

def get_lgb_pipeline():
    params = {
        'learning_rate': 0.0186,
        'max_depth': 9,
        'subsample': 0.6876,
        'min_child_weight': 0.8117,
        'reg_lambda': 6.48,
        'reg_alpha': 3.30,
        'n_estimators': 1000,
        'random_state': seed
    }
    return make_pipeline(
        FeatureEngineeringPipeline(),
        LGBMClassifier(**params)
    )
