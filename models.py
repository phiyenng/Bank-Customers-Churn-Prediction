import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from category_encoders import CatBoostEncoder, MEstimateEncoder
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import roc_auc_score, make_scorer
from model_prep import load_data, list_features
from src.data.preprocessing import SalaryRounder, AgeRounder
from src.data.feature_engineering import FeatureGenerator, Vectorizer
try:
    import tensorflow as tf
except Exception:  # optional dependency for deterministic ops/seed
    tf = None

# Cross-Validation
def cross_val_score(estimator,
    label='', include_original=True, show_importance=False, add_reverse=False, seed=42):
    
    train, test, orig_train = load_data()
    X = train.copy()
    y = X.pop('Exited')

    orig_comp_combo = train.merge(orig_train, on = list(test), how = 'left')
    orig_comp_combo.index = train.index

    orig_test_combo = test.merge(orig_train, on = list(test), how = 'left')
    orig_test_combo.index = test.index

    seed = 42
    splits = 30
    skf = StratifiedKFold(n_splits = splits, random_state = seed, shuffle = True)
    if tf is not None:
        try:
            tf.keras.utils.set_random_seed(seed)
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    # initiate prediction arrays and score lists
    val_predictions = np.zeros((len(X)))
    train_scores, val_scores = [], []
    feature_importances_table = pd.DataFrame({'value': 0}, index=list(X.columns))
    test_predictions = np.zeros((len(test)))

    # training model, predicting prognosis probability, and evaluating metrics
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        model = clone(estimator)

        # define train set
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)

        # define validation set
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        if include_original:
            X_train = pd.concat([orig_train.drop('Exited', axis=1), X_train]).reset_index(drop=True)
            y_train = pd.concat([orig_train.Exited, y_train]).reset_index(drop=True)
        
        if add_reverse:
            X_train = pd.concat([X_train, X_train.iloc[::-1]]).reset_index(drop=True)
            y_train = pd.concat([y_train, y_train.iloc[::-1]]).reset_index(drop=True)
        
        # train model
        model.fit(X_train, y_train)
        
        # make predictions
        train_preds = model.predict_proba(X_train)[:, 1]
        val_preds = model.predict_proba(X_val)[:, 1]
        
        val_predictions[val_idx] += val_preds
        test_predictions += model.predict_proba(test)[:, 1] / skf.get_n_splits()
        
        if show_importance:
            feature_importances_table['value'] += permutation_importance(
                model, X_val, y_val, random_state=seed, 
                scoring=make_scorer(roc_auc_score, needs_proba=True), n_repeats=5
            ).importances_mean / skf.get_n_splits()
        
        # evaluate model for a fold
        train_score = roc_auc_score(y_train, train_preds)
        val_score = roc_auc_score(y_val, val_preds)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    if show_importance:
        plt.figure(figsize=(20, 30))
        plt.title(f'Features with Biggest Importance of {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} Model', size=25, weight='bold')
        sns.barplot(feature_importances_table.sort_values('value', ascending=False).T, orient='h', palette='viridis')
        plt.show()
    else:
        print(f'Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Train Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')
    
    # Post-process predictions if combos are provided
    val_predictions = np.where(orig_comp_combo.Exited_y == 1, 0, np.where(orig_comp_combo.Exited_y == 0, 1, val_predictions))
    test_predictions = np.where(orig_test_combo.Exited == 1, 0, np.where(orig_test_combo.Exited == 0, 1, test_predictions))
    
    return val_scores, val_predictions, test_predictions

# # TensorFlow
# class TensorFlower(BaseEstimator, ClassifierMixin):
#     score_list, oof_list, predict_list, cat_features = list_features()
#     def fit(self, x, y):
#         inputs = tf.keras.Input(shape=(x.shape[1],))
#         inputs_norm = tf.keras.layers.BatchNormalization()(inputs)

#         z = tf.keras.layers.Dense(32)(inputs_norm)
#         z = tf.keras.layers.BatchNormalization()(z)
#         z = tf.keras.layers.LeakyReLU()(z)

#         z = tf.keras.layers.Dense(64)(z)
#         z = tf.keras.layers.BatchNormalization()(z)
#         z = tf.keras.layers.LeakyReLU()(z)

#         z = tf.keras.layers.Dense(16)(z)
#         z = tf.keras.layers.BatchNormalization()(z)
#         z = tf.keras.layers.LeakyReLU()(z)

#         z = tf.keras.layers.Dense(4)(z)
#         z = tf.keras.layers.BatchNormalization()(z)
#         z = tf.keras.layers.LeakyReLU()(z)

#         z = tf.keras.layers.Dense(1)(z)
#         z = tf.keras.layers.BatchNormalization()(z)
#         outputs = tf.keras.activations.sigmoid(z)

#         self.model = tf.keras.Model(inputs, outputs)
#         self.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.AdamW(1e-4))
        
#         self.model.fit(x.to_numpy(), y, epochs=10, verbose=0)
#         self.classes_ = np.unique(y)

#         return self
    
#     def predict_proba(self, x):
#         predictions = np.zeros((len(x), 2))
#         predictions[:, 1] = self.model.predict(x, verbose=0)[:, 0]
#         predictions[:, 0] = 1 - predictions[:, 1]
#         return predictions
    
#     def predict(self, x):
#         return np.argmax(self.predict_proba(x), axis=1)

# def get_tensorflow_pipeline(cat_features):
#     return make_pipeline(
#         SalaryRounder,
#         AgeRounder,
#         FeatureGenerator,
#         CatBoostEncoder(cols=cat_features),
#         TensorFlower()
#     )

# #LightGBM
# def lgb_objective(trial):
#     params = {
#         'learning_rate': trial.suggest_float('learning_rate', .001, .1, log=True),
#         'max_depth': trial.suggest_int('max_depth', 2, 20),
#         'subsample': trial.suggest_float('subsample', .5, 1),
#         'min_child_weight': trial.suggest_float('min_child_weight', .1, 15, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', .1, 20, log=True),
#         'reg_alpha': trial.suggest_float('reg_alpha', .1, 10, log=True),
#         'n_estimators': 1000,
#         'random_state': 42,
#     }

#     model = get_lightgbm_pipeline(params)
#     from sklearn.model_selection import cross_val_score as sk_cv_score
#     score = sk_cv_score(model, X, y, cv=skf, scoring='roc_auc').mean()
#     return score

# lgb_params = {'learning_rate': 0.01864960338160943, 'max_depth': 9,
#                'subsample': 0.6876252164703066, 'min_child_weight': 0.8117588782708633,
#                  'reg_lambda': 6.479178739677389, 'reg_alpha': 3.2952573115561234}

# def get_lightgbm_pipeline(lgb_params):
#     return make_pipeline(
#         SalaryRounder,
#         AgeRounder,
#         FeatureGenerator,
#         Vectorizer(cols=['Surname', 'AllCat'], max_features=1000, n_components=3),
#         CatBoostEncoder(cols=['Surname', 'AllCat', 'CreditScore', 'Age']),
#         MEstimateEncoder(cols=['Geography', 'Gender', 'NumOfProducts']),
#         StandardScaler(),
#         LGBMClassifier(lgb_params=lgb_params)
#     )

# Logistic Regression
def get_logistic_pipeline(seed, cat_features):
    return make_pipeline(
        SalaryRounder,
        AgeRounder,
        FeatureGenerator,
        Vectorizer(cols = ['Surname', 'AllCat', 'EstimatedSalary', 'CreditScore'], max_features=500, n_components=4),
        CatBoostEncoder(cols=cat_features + [f'SurnameSVD{i}' for i in range(4)]),
        StandardScaler(),
        LogisticRegression(random_state=seed, max_iter=1000000000)
    )

# XGBoost
def xgb_objective(trial):
    params = {
        'eta': trial.suggest_float('eta', .001, .3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 30),
        'subsample': trial.suggest_float('subsample', .5, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', .1, 1),
        'min_child_weight': trial.suggest_float('min_child_weight', .1, 20, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', .01, 20, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', .01, 10, log=True),
        'n_estimators': 1000,
        'random_state': 42,
        'tree_method': 'hist',
    }

    optuna_model = make_pipeline(
        SalaryRounder,
        AgeRounder,
        FeatureGenerator,
        Vectorizer(cols = ['Surname', 'AllCat', 'EstimatedSalary', 'CustomerId'], max_features=1000, n_components=3),
        CatBoostEncoder(cols = ['CustomerId', 'Surname', 'EstimatedSalary', 'AllCat', 'CreditScore']),
        MEstimateEncoder(cols = ['Geography', 'Gender']),
        XGBClassifier(**params)
    )
    
    optuna_score, _, _ = cross_val_score(optuna_model)

    return np.mean(optuna_score)

def get_xgboost_pipeline(params):
    return make_pipeline(
        SalaryRounder,
        AgeRounder,
        FeatureGenerator,
        Vectorizer(cols=['Surname', 'AllCat', 'EstimatedSalary', 'CustomerId'], max_features=1000, n_components=3),
        CatBoostEncoder(cols=['CustomerId', 'Surname', 'EstimatedSalary', 'AllCat', 'CreditScore']),
        MEstimateEncoder(cols=['Geography', 'Gender']),
        XGBClassifier(**params)
    )