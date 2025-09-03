from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier
from modules.feature_engineering import FeatureEngineeringPipeline
from models.utils import cross_val_score

seed = 42

def get_cb_pipeline():
    return make_pipeline(
        FeatureEngineeringPipeline(),
        CatBoostClassifier(
            random_state=seed,
            verbose=0,
            n_estimators=1000
        )
    )
