"""
Feature Engineering Module
===========================

This module contains advanced feature engineering techniques specifically
designed for gradient boosting models (XGBoost, LightGBM, CatBoost).

"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from category_encoders import CatBoostEncoder, TargetEncoder
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering for bank customer churn prediction.
    
    This class creates sophisticated features that are particularly effective
    for gradient boosting models.
    """
    
    def __init__(self, create_interactions: bool = True, create_ratios: bool = True,
                 create_aggregations: bool = True, create_binning: bool = True):
        """
        Initialize the feature engineer.
        
        Args:
            create_interactions (bool): Whether to create interaction features
            create_ratios (bool): Whether to create ratio features
            create_aggregations (bool): Whether to create aggregation features
            create_binning (bool): Whether to create binned features
        """
        self.create_interactions = create_interactions
        self.create_ratios = create_ratios
        self.create_aggregations = create_aggregations
        self.create_binning = create_binning
        self.fitted_features = []
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the feature engineer (learn binning thresholds, etc.)."""
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Learn binning thresholds
        if self.create_binning:
            self.binning_thresholds = {}
            for col in self.numerical_cols:
                if col in ['Age', 'CreditScore', 'EstimatedSalary', 'Balance']:
                    self.binning_thresholds[col] = np.percentile(X[col], [25, 50, 75])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input dataframe with engineered features."""
        X_transformed = X.copy()
        
        # 1. Create interaction features
        if self.create_interactions:
            X_transformed = self._create_interaction_features(X_transformed)
        
        # 2. Create ratio features
        if self.create_ratios:
            X_transformed = self._create_ratio_features(X_transformed)
        
        # 3. Create aggregation features
        if self.create_aggregations:
            X_transformed = self._create_aggregation_features(X_transformed)
        
        # 4. Create binned features
        if self.create_binning:
            X_transformed = self._create_binned_features(X_transformed)
        
        # 5. Create domain-specific features
        X_transformed = self._create_domain_features(X_transformed)
        
        return X_transformed
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables."""
        # Age-related interactions
        if 'Age' in X.columns and 'NumOfProducts' in X.columns:
            X['Age_NumProducts_Interaction'] = X['Age'] * X['NumOfProducts']
        
        if 'Age' in X.columns and 'IsActiveMember' in X.columns:
            X['Age_Activity_Interaction'] = X['Age'] * X['IsActiveMember']
        
        # Balance-related interactions
        if 'Balance' in X.columns and 'NumOfProducts' in X.columns:
            X['Balance_NumProducts_Interaction'] = X['Balance'] * X['NumOfProducts']
        
        if 'Balance' in X.columns and 'IsActiveMember' in X.columns:
            X['Balance_Activity_Interaction'] = X['Balance'] * X['IsActiveMember']
        
        # Credit-related interactions
        if 'CreditScore' in X.columns and 'Age' in X.columns:
            X['CreditScore_Age_Interaction'] = X['CreditScore'] * X['Age']
        
        return X
    
    def _create_ratio_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create meaningful ratio features."""
        # Balance to Salary ratio
        if 'Balance' in X.columns and 'EstimatedSalary' in X.columns:
            X['Balance_to_Salary_Ratio'] = X['Balance'] / (X['EstimatedSalary'] + 1)
        
        # Credit Score to Age ratio
        if 'CreditScore' in X.columns and 'Age' in X.columns:
            X['CreditScore_per_Age'] = X['CreditScore'] / X['Age']
        
        # Products per Tenure
        if 'NumOfProducts' in X.columns and 'Tenure' in X.columns:
            X['Products_per_Tenure'] = X['NumOfProducts'] / (X['Tenure'] + 1)
        
        # Salary per Age (earning power)
        if 'EstimatedSalary' in X.columns and 'Age' in X.columns:
            X['Salary_per_Age'] = X['EstimatedSalary'] / X['Age']
        
        return X
    
    def _create_aggregation_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create aggregation features based on categorical variables."""
        # Geography-based aggregations
        if 'Geography' in X.columns:
            for num_col in ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']:
                if num_col in X.columns:
                    geo_stats = X.groupby('Geography')[num_col].agg(['mean', 'std']).reset_index()
                    geo_stats.columns = ['Geography', f'{num_col}_Geography_Mean', f'{num_col}_Geography_Std']
                    X = X.merge(geo_stats, on='Geography', how='left')
                    
                    # Create deviation features
                    X[f'{num_col}_Geography_Deviation'] = X[num_col] - X[f'{num_col}_Geography_Mean']
        
        # Gender-based aggregations
        if 'Gender' in X.columns:
            for num_col in ['Age', 'CreditScore', 'Balance']:
                if num_col in X.columns:
                    gender_stats = X.groupby('Gender')[num_col].agg(['mean']).reset_index()
                    gender_stats.columns = ['Gender', f'{num_col}_Gender_Mean']
                    X = X.merge(gender_stats, on='Gender', how='left')
                    
                    X[f'{num_col}_Gender_Deviation'] = X[num_col] - X[f'{num_col}_Gender_Mean']
        
        return X
    
    def _create_binned_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create binned versions of continuous features."""
        if hasattr(self, 'binning_thresholds'):
            for col, thresholds in self.binning_thresholds.items():
                if col in X.columns:
                    X[f'{col}_Binned'] = pd.cut(X[col], 
                                              bins=[-np.inf] + list(thresholds) + [np.inf],
                                              labels=['Low', 'Medium_Low', 'Medium_High', 'High'])
        
        return X
    
    def _create_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for banking."""
        # Customer value score
        if all(col in X.columns for col in ['Balance', 'EstimatedSalary', 'NumOfProducts']):
            X['Customer_Value_Score'] = (
                0.4 * X['Balance'] / X['Balance'].max() +
                0.3 * X['EstimatedSalary'] / X['EstimatedSalary'].max() +
                0.3 * X['NumOfProducts'] / X['NumOfProducts'].max()
            )
        
        # Risk flags
        if 'Age' in X.columns:
            X['Is_Senior'] = (X['Age'] >= 60).astype(int)
            X['Is_Young'] = (X['Age'] <= 30).astype(int)
        
        if 'Balance' in X.columns:
            X['Has_Zero_Balance'] = (X['Balance'] == 0).astype(int)
            X['Has_High_Balance'] = (X['Balance'] > X['Balance'].quantile(0.9)).astype(int)
        
        if 'CreditScore' in X.columns:
            X['Has_Poor_Credit'] = (X['CreditScore'] < 600).astype(int)
            X['Has_Excellent_Credit'] = (X['CreditScore'] > 800).astype(int)
        
        if 'NumOfProducts' in X.columns:
            X['Has_Single_Product'] = (X['NumOfProducts'] == 1).astype(int)
            X['Has_Multiple_Products'] = (X['NumOfProducts'] > 2).astype(int)
        
        # Engagement features
        if all(col in X.columns for col in ['IsActiveMember', 'HasCrCard', 'Tenure']):
            X['Engagement_Score'] = X['IsActiveMember'] + X['HasCrCard'] + (X['Tenure'] > 5).astype(int)
        
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Enhanced categorical encoding optimized for gradient boosting models.
    """
    
    def __init__(self, encoding_method: str = 'catboost', handle_unknown: str = 'value'):
        """
        Initialize categorical encoder.
        
        Args:
            encoding_method (str): Encoding method ('catboost', 'target', 'label')
            handle_unknown (str): How to handle unknown categories
        """
        self.encoding_method = encoding_method
        self.handle_unknown = handle_unknown
        self.encoders = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the encoders."""
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if self.encoding_method == 'catboost':
                encoder = CatBoostEncoder(handle_unknown=self.handle_unknown)
            elif self.encoding_method == 'target':
                encoder = TargetEncoder(handle_unknown=self.handle_unknown)
            else:  # label encoding
                encoder = LabelEncoder()
            
            if self.encoding_method in ['catboost', 'target'] and y is not None:
                encoder.fit(X[col], y)
            else:
                encoder.fit(X[col])
            
            self.encoders[col] = encoder
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features."""
        X_transformed = X.copy()
        
        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                X_transformed[col] = encoder.transform(X_transformed[col])
        
        return X_transformed


class FeaturePipeline:
    """
    Complete feature engineering pipeline for gradient boosting models.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature pipeline.
        
        Args:
            config (dict): Configuration dictionary for feature engineering
        """
        default_config = {
            'create_interactions': True,
            'create_ratios': True,
            'create_aggregations': True,
            'create_binning': True,
            'encoding_method': 'catboost',
            'scale_features': False  # Usually not needed for tree-based models
        }
        
        self.config = {**default_config, **(config or {})}
        self.feature_engineer = None
        self.categorical_encoder = None
        self.scaler = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeaturePipeline':
        """Fit the complete pipeline."""
        # 1. Feature Engineering
        self.feature_engineer = AdvancedFeatureEngineer(
            create_interactions=self.config['create_interactions'],
            create_ratios=self.config['create_ratios'],
            create_aggregations=self.config['create_aggregations'],
            create_binning=self.config['create_binning']
        )
        X_engineered = self.feature_engineer.fit_transform(X, y)
        
        # 2. Categorical Encoding
        self.categorical_encoder = CategoricalEncoder(
            encoding_method=self.config['encoding_method']
        )
        X_encoded = self.categorical_encoder.fit_transform(X_engineered, y)
        
        # 3. Scaling (optional for tree-based models)
        if self.config['scale_features']:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_encoded)
            X_final = pd.DataFrame(X_scaled, columns=X_encoded.columns, index=X_encoded.index)
        else:
            X_final = X_encoded
        
        self.feature_names = list(X_final.columns)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline."""
        # 1. Feature Engineering
        X_engineered = self.feature_engineer.transform(X)
        
        # 2. Categorical Encoding
        X_encoded = self.categorical_encoder.transform(X_engineered)
        
        # 3. Scaling (if enabled)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_encoded)
            X_final = pd.DataFrame(X_scaled, columns=X_encoded.columns, index=X_encoded.index)
        else:
            X_final = X_encoded
        
        return X_final
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after transformation."""
        return self.feature_names if self.feature_names else []
    
    def get_feature_importance_analysis(self, X: pd.DataFrame, y: pd.Series, 
                                      model_type: str = 'xgboost') -> Dict:
        """
        Analyze feature importance using the specified model.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target variable
            model_type (str): Type of model to use ('xgboost', 'lightgbm', 'catboost')
            
        Returns:
            Dict: Feature importance analysis
        """
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        
        # Transform features
        X_transformed = self.fit_transform(X, y)
        
        # Select model
        if model_type == 'xgboost':
            model = XGBClassifier(random_state=42, eval_metric='logloss')
        elif model_type == 'lightgbm':
            model = LGBMClassifier(random_state=42, verbose=-1)
        else:  # catboost
            model = CatBoostClassifier(random_state=42, verbose=False)
        
        # Fit model and get feature importance
        model.fit(X_transformed, y)
        
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        else:
            importance_scores = model.get_feature_importance()
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.get_feature_names(),
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return {
            'feature_importance': feature_importance,
            'top_10_features': feature_importance.head(10)['feature'].tolist(),
            'model_score': model.score(X_transformed, y)
        }


if __name__ == "__main__":
    # Example usage
    print("=== Feature Engineering Module Test ===")
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Age': np.random.randint(18, 80, 1000),
        'CreditScore': np.random.randint(350, 850, 1000),
        'Balance': np.random.uniform(0, 250000, 1000),
        'EstimatedSalary': np.random.uniform(20000, 200000, 1000),
        'NumOfProducts': np.random.randint(1, 5, 1000),
        'Tenure': np.random.randint(0, 11, 1000),
        'IsActiveMember': np.random.choice([0, 1], 1000),
        'HasCrCard': np.random.choice([0, 1], 1000),
        'Geography': np.random.choice(['France', 'Germany', 'Spain'], 1000),
        'Gender': np.random.choice(['Male', 'Female'], 1000),
        'Exited': np.random.choice([0, 1], 1000)
    })
    
    X = sample_data.drop('Exited', axis=1)
    y = sample_data['Exited']
    
    # Test feature pipeline
    pipeline = FeaturePipeline()
    X_transformed = pipeline.fit_transform(X, y)
    
    print(f"Original features: {X.shape[1]}")
    print(f"Transformed features: {X_transformed.shape[1]}")
    print(f"New features created: {X_transformed.shape[1] - X.shape[1]}")
    
    # Show feature importance analysis
    importance_analysis = pipeline.get_feature_importance_analysis(X, y)
    print(f"\nTop 10 most important features:")
    for i, feature in enumerate(importance_analysis['top_10_features'], 1):
        print(f"{i:2d}. {feature}")