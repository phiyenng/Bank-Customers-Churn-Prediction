"""
Feature Engineering Module
==========================

This module handles feature creation, transformation, selection, and reduction
for the bank customer churn prediction project.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer, StandardScaler, QuantileTransformer, MinMaxScaler
from sklearn.feature_selection import (
    VarianceThreshold, 
    SelectKBest, 
    SelectPercentile,
    mutual_info_classif, 
    mutual_info_regression,
    f_classif,
    f_regression
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import mutual_info_score

# =============================================================================
# FEATURE CREATION FUNCTIONS
# =============================================================================

def add_age_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create age categories by binning the Age column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing 'Age' column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added 'Age_Category' column
    """
    df = df.copy()
    df['Age_Category'] = pd.cut(
        df['Age'], 
        bins=[18, 30, 40, 50, 60, 100], 
        labels=['18-30', '30-40', '40-50', '50-60', '60+'],
        include_lowest=True
    )
    return df


def add_credit_score_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create credit score ranges by binning the CreditScore column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing 'CreditScore' column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added 'Credit_Score_Range' column
    """
    df = df.copy()
    df['Credit_Score_Range'] = pd.cut(
        df['CreditScore'], 
        bins=[0, 300, 600, 700, 800, 900], 
        labels=['0-300', '300-600', '600-700', '700-800', '900+'],
        include_lowest=True
    )
    return df


def add_balance_salary_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create balance to salary ratio feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing 'Balance' and 'EstimatedSalary' columns
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added 'Balance_Salary_Ratio' column
    """
    df = df.copy()
    epsilon = 1e-6  # Small value to avoid division by zero
    df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + epsilon)
    return df


def add_geo_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create geography and gender interaction feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing 'Geography' and 'Gender' columns
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added 'Geo_Gender' column
    """
    df = df.copy()
    df['Geo_Gender'] = df['Geography'] + '_' + df['Gender']
    return df


def add_total_products_used(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create total products used feature by combining NumOfProducts and HasCrCard.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing 'NumOfProducts' and 'HasCrCard' columns
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added 'Total_Products_Used' column
    """
    df = df.copy()
    df['Total_Products_Used'] = df['NumOfProducts'] + df['HasCrCard']
    return df


def add_tp_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create total products and gender interaction feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing 'Total_Products_Used' and 'Gender' columns
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added 'Tp_Gender' column
    """
    df = df.copy()
    df['Tp_Gender'] = df['Total_Products_Used'].astype(str) + '_' + df['Gender']
    return df


# =============================================================================
# FEATURE TRANSFORMATION FUNCTIONS
# =============================================================================

class Transformation:
    """
    Optimized feature-wise transformations for Gradient Boosting models.
    Includes optional OneHot encoding for categorical columns.
    """
    
    def __init__(self, method: str = 'standard', handle_categorical: bool = True):
        # 'method' retained for backward compatibility; not used in optimized mapping
        self.method = method
        self.handle_categorical = handle_categorical
        self.encoder = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.fitted = False
        
    def fit(self, df: pd.DataFrame, numeric: Optional[List[str]] = None, categorical: Optional[List[str]] = None):
        """
        Fit transformation parameters on training data.
        """
        if numeric is None:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if categorical is None and self.handle_categorical:
            categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
        else:
            categorical = []

        self.numeric_cols = numeric
        self.categorical_cols = categorical

        # Optimized per-feature transformers for gradient boosting models
        self.optimized_transformers = {}
        for col in numeric:
            if col == 'Age':
                # PowerTransformer (yeo-johnson) + StandardScaler
                pt = PowerTransformer(method='yeo-johnson', standardize=False)
                ss = StandardScaler()
                self.optimized_transformers[col] = ('power_standard', (pt, ss))
                pt.fit(df[[col]])
                ss.fit(pt.transform(df[[col]]))

            elif col == 'Balance':
                # QuantileTransformer to uniform
                qt = QuantileTransformer(n_quantiles=min(1000, len(df)), output_distribution='uniform', random_state=42)
                self.optimized_transformers[col] = ('quantile_uniform', (qt,))
                qt.fit(df[[col]])

            elif col == 'CreditScore':
                # Box-Cox (requires positive), shift if needed, then StandardScaler
                shift = 0.0
                series = df[col]
                if (series <= 0).any():
                    shift = abs(series.min()) + 1e-6
                pt = PowerTransformer(method='box-cox', standardize=False)
                ss = StandardScaler()
                self.optimized_transformers[col] = ('boxcox_standard', (pt, ss, shift))
                pt.fit((series + shift).values.reshape(-1,1))
                ss.fit(pt.transform((series + shift).values.reshape(-1,1)))

            elif col == 'EstimatedSalary':
                # QuantileTransformer (uniform) + StandardScaler
                qt = QuantileTransformer(n_quantiles=min(1000, len(df)), output_distribution='uniform', random_state=42)
                ss = StandardScaler()
                self.optimized_transformers[col] = ('quantile_standard', (qt, ss))
                qt.fit(df[[col]])
                ss.fit(qt.transform(df[[col]]))

            elif col == 'Tenure':
                # MinMaxScaler to range (-2, 2)
                mm = MinMaxScaler(feature_range=(-2, 2))
                self.optimized_transformers[col] = ('minmax_custom', (mm,))
                mm.fit(df[[col]])

            elif col == 'NumOfProducts':
                # MinMaxScaler + StandardScaler
                mm = MinMaxScaler()
                ss = StandardScaler()
                self.optimized_transformers[col] = ('minmax_standard', (mm, ss))
                mm.fit(df[[col]])
                ss.fit(mm.transform(df[[col]]))

            elif col == 'HasCrCard':
                # MinMaxScaler to (-1, 1)
                mm = MinMaxScaler(feature_range=(-1, 1))
                self.optimized_transformers[col] = ('minmax_neg1_1', (mm,))
                mm.fit(df[[col]])

            elif col == 'IsActiveMember':
                # Preserve as-is (important negative correlation per request)
                self.optimized_transformers[col] = ('none', tuple())
            else:
                # Pass-through for unspecified columns
                self.optimized_transformers[col] = ('none', tuple())

        # Fit encoder for categorical
        if self.handle_categorical and categorical:
            self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown="ignore")
            self.encoder.fit(df[categorical])

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to the dataframe.
        """
        if not self.fitted:
            raise ValueError("Call fit() before transform().")

        df_transformed = pd.DataFrame(index=df.index)

        # --- Numeric scaling ---
        for col in self.numeric_cols:
            if col not in df.columns: 
                continue

            if col in getattr(self, 'optimized_transformers', {}):
                mode, objs = self.optimized_transformers[col]

                if mode == 'power_standard':
                    pt, ss = objs
                    val = pt.transform(df[[col]])
                    df_transformed[col] = ss.transform(val)

                elif mode == 'quantile_uniform':
                    (qt,) = objs
                    df_transformed[col] = qt.transform(df[[col]])

                elif mode == 'boxcox_standard':
                    pt, ss, shift = objs
                    arr = (df[col] + shift).values.reshape(-1,1)
                    val = pt.transform(arr)
                    df_transformed[col] = ss.transform(val)

                elif mode == 'quantile_standard':
                    qt, ss = objs
                    val = qt.transform(df[[col]])
                    df_transformed[col] = ss.transform(val)

                elif mode == 'minmax_custom':
                    (mm,) = objs
                    df_transformed[col] = mm.transform(df[[col]])

                elif mode == 'minmax_standard':
                    mm, ss = objs
                    val = mm.transform(df[[col]])
                    df_transformed[col] = ss.transform(val)

                elif mode == 'minmax_neg1_1':
                    (mm,) = objs
                    df_transformed[col] = mm.transform(df[[col]])

                elif mode == 'none':
                    df_transformed[col] = df[col]
                else:
                    df_transformed[col] = df[col]
            else:
                # fallback already set to 'none' in fit
                df_transformed[col] = df[col]

        # --- Categorical encoding ---
        if self.handle_categorical and self.categorical_cols:
            encoded = self.encoder.transform(df[self.categorical_cols])
            encoded_df = pd.DataFrame(encoded, 
                                      columns=self.encoder.get_feature_names_out(self.categorical_cols),
                                      index=df.index)
            df_transformed = pd.concat([df_transformed, encoded_df], axis=1)

        return df_transformed

    def fit_transform(self, df: pd.DataFrame, numeric: Optional[List[str]] = None, categorical: Optional[List[str]] = None) -> pd.DataFrame:
        return self.fit(df, numeric, categorical).transform(df)


# =============================================================================
# FEATURE SELECTION & REDUCTION (PLACEHOLDERS)
# =============================================================================

def select_features(df: pd.DataFrame, method: str = 'correlation', threshold: float = 0.95, 
                   target_col: Optional[str] = None, k: Optional[int] = None, 
                   percentile: Optional[int] = None) -> pd.DataFrame:
    """
    Select relevant features based on specified method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str, default='correlation'
        Selection method ('correlation', 'variance', 'mutual_info', 'f_score')
    threshold : float, default=0.95
        Threshold for correlation-based selection
    target_col : str, optional
        Target column name for supervised methods
    k : int, optional
        Number of top features to select (for mutual_info and f_score)
    percentile : int, optional
        Percentile of features to select (for mutual_info and f_score)
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with selected features
    """
    df_selected = df.copy()
    
    if method == 'correlation':
        # Remove highly correlated features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            
            # Find pairs of highly correlated features
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to drop
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            
            print(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
            df_selected = df_selected.drop(columns=to_drop)
    
    elif method == 'variance':
        # Remove low variance features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if numeric_cols:
            var_threshold = VarianceThreshold(threshold=threshold)
            var_threshold.fit(df[numeric_cols])
            
            # Get selected features
            selected_features = var_threshold.get_support()
            features_to_keep = [col for col, keep in zip(numeric_cols, selected_features) if keep]
            features_to_drop = [col for col in numeric_cols if col not in features_to_keep]
            
            print(f"Removing {len(features_to_drop)} low variance features: {features_to_drop}")
            df_selected = df_selected.drop(columns=features_to_drop)
    
    elif method == 'mutual_info':
        if target_col is None:
            raise ValueError("target_col must be specified for mutual_info method")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle categorical features by encoding them
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if categorical_cols:
            # One-hot encode categorical columns
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            encoded_categorical = encoder.fit_transform(X[categorical_cols])
            encoded_df = pd.DataFrame(
                encoded_categorical,
                columns=encoder.get_feature_names_out(categorical_cols),
                index=X.index
            )
            X_processed = pd.concat([X[numeric_cols], encoded_df], axis=1)
        else:
            X_processed = X[numeric_cols]
        
        # Handle categorical target
        if y.dtype == 'object' or y.dtype.name == 'category':
            # Classification
            mi_scores = mutual_info_classif(X_processed, y, random_state=42)
        else:
            # Regression
            mi_scores = mutual_info_regression(X_processed, y, random_state=42)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'mutual_info': mi_scores[:len(feature_cols)]
        }).sort_values('mutual_info', ascending=False)
        
        # Select features
        if k is not None:
            selected_features = feature_importance.head(k)['feature'].tolist()
        elif percentile is not None:
            n_features = int(len(feature_cols) * percentile / 100)
            selected_features = feature_importance.head(n_features)['feature'].tolist()
        else:
            # Default: select top 50% of features
            n_features = max(1, len(feature_cols) // 2)
            selected_features = feature_importance.head(n_features)['feature'].tolist()
        
        features_to_drop = [col for col in feature_cols if col not in selected_features]
        print(f"Selected {len(selected_features)} features based on mutual information")
        print(f"Top features: {selected_features[:5]}")
        df_selected = df_selected.drop(columns=features_to_drop)
    
    elif method == 'f_score':
        if target_col is None:
            raise ValueError("target_col must be specified for f_score method")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle categorical features by encoding them
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if categorical_cols:
            # One-hot encode categorical columns
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            encoded_categorical = encoder.fit_transform(X[categorical_cols])
            encoded_df = pd.DataFrame(
                encoded_categorical,
                columns=encoder.get_feature_names_out(categorical_cols),
                index=X.index
            )
            X_processed = pd.concat([X[numeric_cols], encoded_df], axis=1)
        else:
            X_processed = X[numeric_cols]
        
        # Handle categorical target
        if y.dtype == 'object' or y.dtype.name == 'category':
            # Classification
            f_scores, _ = f_classif(X_processed, y)
        else:
            # Regression
            f_scores, _ = f_regression(X_processed, y)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'f_score': f_scores[:len(feature_cols)]
        }).sort_values('f_score', ascending=False)
        
        # Select features
        if k is not None:
            selected_features = feature_importance.head(k)['feature'].tolist()
        elif percentile is not None:
            n_features = int(len(feature_cols) * percentile / 100)
            selected_features = feature_importance.head(n_features)['feature'].tolist()
        else:
            # Default: select top 50% of features
            n_features = max(1, len(feature_cols) // 2)
            selected_features = feature_importance.head(n_features)['feature'].tolist()
        
        features_to_drop = [col for col in feature_cols if col not in selected_features]
        print(f"Selected {len(selected_features)} features based on F-score")
        print(f"Top features: {selected_features[:5]}")
        df_selected = df_selected.drop(columns=features_to_drop)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'correlation', 'variance', 'mutual_info', 'f_score'")
    
    return df_selected


def reduce_features(df: pd.DataFrame, method: str = 'pca', n_components: int = 10, 
                   target_col: Optional[str] = None, random_state: int = 42) -> pd.DataFrame:
    """
    Reduce feature dimensionality using specified method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str, default='pca'
        Reduction method ('pca', 'lda', 'tsne')
    n_components : int, default=10
        Number of components to keep
    target_col : str, optional
        Target column name (required for LDA)
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with reduced features
    """
    df_reduced = df.copy()
    
    # Separate features and target if target_col is provided
    if target_col and target_col in df.columns:
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        keep_target = True
    else:
        X = df
        y = None
        keep_target = False
    
    # Handle non-numeric columns by encoding them
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if categorical_cols:
        # One-hot encode categorical columns
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoded_categorical = encoder.fit_transform(X[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_categorical,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=X.index
        )
        X_processed = pd.concat([X[numeric_cols], encoded_df], axis=1)
    else:
        X_processed = X[numeric_cols]
    
    # Apply dimensionality reduction
    if method == 'pca':
        # Principal Component Analysis
        if n_components >= X_processed.shape[1]:
            n_components = min(X_processed.shape[1] - 1, 10)
            print(f"Adjusted n_components to {n_components} (max available: {X_processed.shape[1] - 1})")
        
        pca = PCA(n_components=n_components, random_state=random_state)
        X_reduced = pca.fit_transform(X_processed)
        
        # Create column names
        component_names = [f'PC_{i+1}' for i in range(n_components)]
        
        # Print explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        print(f"PCA - Explained variance ratio: {explained_variance_ratio}")
        print(f"PCA - Cumulative explained variance: {cumulative_variance}")
        print(f"PCA - Total variance explained: {cumulative_variance[-1]:.3f}")
    
    elif method == 'lda':
        # Linear Discriminant Analysis
        if target_col is None or y is None:
            raise ValueError("target_col must be specified for LDA method")
        
        # LDA requires at least 2 classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError("LDA requires at least 2 classes in target variable")
        
        # LDA can have at most (n_classes - 1) components
        max_components = len(unique_classes) - 1
        if n_components > max_components:
            n_components = max_components
            print(f"Adjusted n_components to {n_components} (max for LDA: {max_components})")
        
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_reduced = lda.fit_transform(X_processed, y)
        
        # Create column names
        component_names = [f'LD_{i+1}' for i in range(n_components)]
        
        # Print explained variance
        explained_variance_ratio = lda.explained_variance_ratio_
        print(f"LDA - Explained variance ratio: {explained_variance_ratio}")
        print(f"LDA - Total variance explained: {np.sum(explained_variance_ratio):.3f}")
    
    elif method == 'tsne':
        # t-SNE (t-Distributed Stochastic Neighbor Embedding)
        if n_components > 3:
            n_components = 3
            print(f"Adjusted n_components to {n_components} (t-SNE typically uses 2-3 components)")
        
        # t-SNE is computationally expensive, so we might want to limit the number of samples
        max_samples = 1000
        if X_processed.shape[0] > max_samples:
            print(f"t-SNE is computationally expensive. Using {max_samples} samples for demonstration.")
            # Sample data for t-SNE
            sample_indices = np.random.choice(X_processed.shape[0], max_samples, replace=False)
            X_sample = X_processed.iloc[sample_indices]
            if y is not None:
                y_sample = y.iloc[sample_indices]
            else:
                y_sample = None
        else:
            X_sample = X_processed
            y_sample = y
        
        tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=30)
        X_reduced = tsne.fit_transform(X_sample)
        
        # Create column names
        component_names = [f'tSNE_{i+1}' for i in range(n_components)]
        
        print(f"t-SNE - Reduced {X_sample.shape[1]} features to {n_components} components")
        print("Note: t-SNE is non-linear and primarily for visualization")
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'pca', 'lda', 'tsne'")
    
    # Create reduced dataframe
    reduced_df = pd.DataFrame(X_reduced, columns=component_names, index=X_processed.index)
    
    # Add target column back if it was provided
    if keep_target and target_col:
        reduced_df[target_col] = y
    
    print(f"Feature reduction completed: {X_processed.shape[1]} features -> {n_components} components")
    
    return reduced_df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def apply_all_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in the correct order.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with all engineered features
    """
    df = add_age_category(df)
    df = add_credit_score_range(df)
    df = add_balance_salary_ratio(df)
    df = add_geo_gender(df)
    df = add_total_products_used(df)
    df = add_tp_gender(df)
    
    return df


def get_feature_engineering_pipeline() -> List[str]:
    """
    Get the list of feature engineering functions in order.
    
    Returns:
    --------
    List[str]
        List of function names in execution order
    """
    return [
        'add_age_category',
        'add_credit_score_range', 
        'add_balance_salary_ratio',
        'add_geo_gender',
        'add_total_products_used',
        'add_tp_gender'
    ]
