"""
Feature Engineering Module
==========================

This module handles feature creation, transformation, selection.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler, QuantileTransformer, MinMaxScaler
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
    f_classif,
    f_regression
)

# =============================================================================
# FEATURE CREATION FUNCTIONS
# =============================================================================
class FeatureCreation:

    @staticmethod
    def add_age_category(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Age_Category'] = pd.cut(
            df['Age'],
            bins=[18, 30, 40, 50, 60, 100],
            labels=['18-30', '30-40', '40-50', '50-60', '60+'],
            include_lowest=True
        )
        return df

    @staticmethod
    def add_credit_score_range(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Credit_Score_Range'] = pd.cut(
            df['CreditScore'],
            bins=[0, 300, 600, 700, 800, 900],
            labels=['0-300', '300-600', '600-700', '700-800', '900+'],
            include_lowest=True
        )
        return df

    @staticmethod
    def add_balance_salary_ratio(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        epsilon = 1e-6
        df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + epsilon)
        return df

    @staticmethod
    def add_geo_gender(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Geo_Gender'] = df['Geography'] + '_' + df['Gender']
        return df

    @staticmethod
    def add_total_products_used(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Total_Products_Used'] = df['NumOfProducts'] + df['HasCrCard']
        return df

    @staticmethod
    def add_tp_gender(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Tp_Gender'] = df['Total_Products_Used'].astype(str) + '_' + df['Gender']
        return df

    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature creation steps sequentially.
        """
        df_new = df.copy()
        df_new = self.add_age_category(df_new)
        df_new = self.add_credit_score_range(df_new)
        df_new = self.add_balance_salary_ratio(df_new)
        df_new = self.add_geo_gender(df_new)
        df_new = self.add_total_products_used(df_new)
        df_new = self.add_tp_gender(df_new)

        return df_new

# =============================================================================
# FEATURE TRANSFORMATION FUNCTIONS
# =============================================================================

class FeatureTransformation:
    
    def __init__(self, handle_categorical: bool = True):
        self.handle_categorical = handle_categorical
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.optimized_transformers = {}
        self.gender_mapping = None
        self.geo_encoder = None
        self.fitted = False

    def fit(self, df: pd.DataFrame, numeric: Optional[List[str]] = None, categorical: Optional[List[str]] = None):
        """
        Fit transformers for each feature.
        """
        if numeric is None:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if categorical is None and self.handle_categorical:
            categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
        else:
            categorical = []

        self.numeric_cols = numeric
        self.categorical_cols = categorical

        # --- Numeric transformers ---
        for col in numeric:
            series = df[col]

            if col == "Age":
                pt, ss = PowerTransformer(method="yeo-johnson", standardize=False), StandardScaler()
                pt.fit(series.values.reshape(-1, 1))
                ss.fit(pt.transform(series.values.reshape(-1, 1)))
                self.optimized_transformers[col] = ("power_standard", (pt, ss))

            elif col == "Balance":
                qt = QuantileTransformer(
                    n_quantiles=min(1000, len(df)),
                    output_distribution="uniform",
                    random_state=42,
                )
                qt.fit(series.values.reshape(-1, 1))
                self.optimized_transformers[col] = ("quantile_uniform", (qt,))

            elif col == "CreditScore":
                shift = abs(series.min()) + 1e-6 if (series <= 0).any() else 0.0
                pt, ss = PowerTransformer(method="box-cox", standardize=False), StandardScaler()
                pt.fit((series + shift).values.reshape(-1, 1))
                ss.fit(pt.transform((series + shift).values.reshape(-1, 1)))
                self.optimized_transformers[col] = ("boxcox_standard", (pt, ss, shift))

            elif col == "EstimatedSalary":
                qt, ss = QuantileTransformer(
                    n_quantiles=min(1000, len(df)),
                    output_distribution="uniform",
                    random_state=42,
                ), StandardScaler()
                qt.fit(series.values.reshape(-1, 1))
                ss.fit(qt.transform(series.values.reshape(-1, 1)))
                self.optimized_transformers[col] = ("quantile_standard", (qt, ss))

            elif col == "Tenure":
                mm = MinMaxScaler(feature_range=(-2, 2))
                mm.fit(series.values.reshape(-1, 1))
                self.optimized_transformers[col] = ("minmax_custom", (mm,))

            elif col == "NumOfProducts":
                mm, ss = MinMaxScaler(), StandardScaler()
                mm.fit(series.values.reshape(-1, 1))
                ss.fit(mm.transform(series.values.reshape(-1, 1)))
                self.optimized_transformers[col] = ("minmax_standard", (mm, ss))

            elif col == "HasCrCard":
                mm = MinMaxScaler(feature_range=(-1, 1))
                mm.fit(series.values.reshape(-1, 1))
                self.optimized_transformers[col] = ("minmax_neg1_1", (mm,))

            elif col == "IsActiveMember":
                self.optimized_transformers[col] = ("none", ())

            else:
                self.optimized_transformers[col] = ("none", ())

        # --- Categorical encoders ---
        if self.handle_categorical:
            if "Gender" in categorical:
                self.gender_mapping = {"Female": 0, "Male": 1}
                categorical.remove("Gender")

            if "Geography" in categorical:
                self.geo_encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
                self.geo_encoder.fit(df[["Geography"]])
                categorical.remove("Geography")

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformations.
        """
        if not self.fitted:
            raise ValueError("Call fit() before transform().")

        df_out = pd.DataFrame(index=df.index)

        # --- Numeric ---
        for col in self.numeric_cols:
            if col not in df.columns:
                continue

            mode, objs = self.optimized_transformers[col]

            if mode == "power_standard":
                pt, ss = objs
                df_out[col] = ss.transform(pt.transform(df[[col]]))

            elif mode == "quantile_uniform":
                (qt,) = objs
                df_out[col] = qt.transform(df[[col]])

            elif mode == "boxcox_standard":
                pt, ss, shift = objs
                arr = (df[col] + shift).values.reshape(-1, 1)
                df_out[col] = ss.transform(pt.transform(arr))

            elif mode == "quantile_standard":
                qt, ss = objs
                df_out[col] = ss.transform(qt.transform(df[[col]]))

            elif mode == "minmax_custom":
                (mm,) = objs
                df_out[col] = mm.transform(df[[col]])

            elif mode == "minmax_standard":
                mm, ss = objs
                df_out[col] = ss.transform(mm.transform(df[[col]]))

            elif mode == "minmax_neg1_1":
                (mm,) = objs
                df_out[col] = mm.transform(df[[col]])

            elif mode == "none":
                df_out[col] = df[col]

        # --- Categorical ---
        if self.handle_categorical:
            if self.gender_mapping and "Gender" in df.columns:
                df_out["Gender"] = df["Gender"].map(self.gender_mapping).fillna(-1).astype(int)

            if self.geo_encoder and "Geography" in df.columns:
                geo_encoded = self.geo_encoder.transform(df[["Geography"]])
                geo_cols = self.geo_encoder.get_feature_names_out(["Geography"])
                df_out = pd.concat([df_out, pd.DataFrame(geo_encoded, columns=geo_cols, index=df.index)], axis=1)

        return df_out

    def fit_transform(self, df: pd.DataFrame, numeric: Optional[List[str]] = None, categorical: Optional[List[str]] = None) -> pd.DataFrame:
        return self.fit(df, numeric, categorical).transform(df)



# =============================================================================
# FEATURE SELECTION
# =============================================================================

class FeatureSelection:

    def __init__(self, method: str = "correlation", threshold: float = 0.95,
                 target_col: Optional[str] = None, k: Optional[int] = None,
                 percentile: Optional[int] = None):
        self.method = method
        self.threshold = threshold
        self.target_col = target_col
        self.k = k
        self.percentile = percentile
        self.selected_features: Optional[List[str]] = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.method == "correlation":
            return self._select_correlation(df)
        elif self.method == "variance":
            return self._select_variance(df)
        elif self.method == "mutual_info":
            return self._select_mutual_info(df)
        elif self.method == "f_score":
            return self._select_f_score(df)
        else:
            raise ValueError(
                f"Unknown method: {self.method}. "
                f"Choose from ['correlation','variance','mutual_info','f_score']"
            )

    # ---------------- Methods ---------------- #

    def _select_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)

        if len(numeric_cols) <= 1:
            return df

        corr_matrix = df[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [c for c in upper_tri.columns if any(upper_tri[c] > self.threshold)]

        print(f"[Correlation] Dropping {len(to_drop)} features: {to_drop}")
        return df.drop(columns=to_drop)

    def _select_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)

        if not numeric_cols:
            return df

        vt = VarianceThreshold(threshold=self.threshold)
        vt.fit(df[numeric_cols])

        keep = [col for col, k in zip(numeric_cols, vt.get_support()) if k]
        drop = [col for col in numeric_cols if col not in keep]

        print(f"[Variance] Dropping {len(drop)} features: {drop}")
        return df.drop(columns=drop)

    def _select_mutual_info(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.target_col is None:
            raise ValueError("target_col must be specified for mutual_info")

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # mutual info
        if y.dtype == "object" or y.dtype.name == "category":
            scores = mutual_info_classif(X, y, random_state=42)
        else:
            scores = mutual_info_regression(X, y, random_state=42)

        return self._keep_top(df, X.columns.tolist(), scores, "Mutual Info")

    def _select_f_score(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.target_col is None:
            raise ValueError("target_col must be specified for f_score")

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # f-score
        if y.dtype == "object" or y.dtype.name == "category":
            scores, _ = f_classif(X, y)
        else:
            scores, _ = f_regression(X, y)

        return self._keep_top(df, X.columns.tolist(), scores, "F-score")

    # ---------------- Helper ---------------- #

    def _keep_top(self, df: pd.DataFrame, features: List[str], scores, method_name: str) -> pd.DataFrame:
        """Keep top-k or percentile features"""
        importance = pd.DataFrame({"feature": features, "score": scores}).sort_values("score", ascending=False)

        if self.k:
            keep = importance.head(self.k)["feature"].tolist()
        elif self.percentile:
            n_keep = int(len(features) * self.percentile / 100)
            keep = importance.head(n_keep)["feature"].tolist()
        else:
            n_keep = max(1, len(features) // 2)
            keep = importance.head(n_keep)["feature"].tolist()

        drop = [f for f in features if f not in keep]
        print(f"[{method_name}] Keeping {len(keep)} features, Dropping {len(drop)} features")
        print(f"Top features: {keep[:5]}")
        return df.drop(columns=drop)
