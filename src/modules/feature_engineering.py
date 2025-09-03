import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, chi2
from category_encoders.target_encoder import TargetEncoder


class NumericalTransformer:
    def __init__(self, method="yeo-johnson"):
        self.method = method
        self.transformer = None

    def fit(self, X: pd.DataFrame):
        if self.method in ["boxcox", "yeo-johnson"]:
            self.transformer = PowerTransformer(method=self.method)
            self.transformer.fit(X)
        elif self.method in ["log", "sqrt"]:
            # no fitting needed
            pass
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        if self.method == "log":
            return np.log1p(X_new)
        elif self.method == "sqrt":
            return np.sqrt(X_new)
        elif self.method in ["boxcox", "yeo-johnson"]:
            return pd.DataFrame(self.transformer.transform(X_new), columns=X_new.columns, index=X_new.index)
        else:
            return X_new


class CategoricalEncoder:
    def __init__(self, method="onehot", target=None):
        self.method = method
        self.target = target
        self.encoder = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        if self.method == "onehot":
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.encoder.fit(X)
        elif self.method == "label":
            self.encoder = {col: LabelEncoder().fit(X[col]) for col in X.columns}
        elif self.method == "target":
            if y is None:
                raise ValueError("Target encoding requires y")
            self.encoder = TargetEncoder()
            self.encoder.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method == "onehot":
            arr = self.encoder.transform(X)
            cols = self.encoder.get_feature_names_out(X.columns)
            return pd.DataFrame(arr, columns=cols, index=X.index)
        elif self.method == "label":
            X_new = X.copy()
            for col, enc in self.encoder.items():
                X_new[col] = enc.transform(X[col])
            return X_new
        elif self.method == "target":
            return self.encoder.transform(X)
        else:
            return X


class ClusterFeatures:
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None

    def fit(self, X: pd.DataFrame):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        clusters = self.kmeans.predict(X)
        return pd.DataFrame({"cluster": clusters}, index=X.index)


class ArithmeticFeatures:
    def __init__(self):
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        cols = X.columns
        if len(cols) >= 2:
            # ratio of first two features
            X_new[f"{cols[0]}_div_{cols[1]}"] = X[cols[0]] / (X[cols[1]] + 1e-6)
            # product
            X_new[f"{cols[0]}_x_{cols[1]}"] = X[cols[0]] * X[cols[1]]
            # difference
            X_new[f"{cols[0]}_minus_{cols[1]}"] = X[cols[0]] - X[cols[1]]
        return X_new


class FeatureReducer:
    def __init__(self, method="pca", n_components=5):
        self.method = method
        self.n_components = n_components
        self.reducer = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        if self.method == "pca":
            self.reducer = PCA(n_components=self.n_components)
            self.reducer.fit(X)
        elif self.method == "svd":
            self.reducer = TruncatedSVD(n_components=self.n_components)
            self.reducer.fit(X)
        elif self.method == "chi2":
            if y is None:
                raise ValueError("Chi2 requires target variable")
            self.reducer = SelectKBest(score_func=chi2, k=self.n_components)
            self.reducer.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self.reducer.transform(X)
        cols = [f"{self.method}_{i}" for i in range(self.n_components)]
        return pd.DataFrame(arr, columns=cols, index=X.index)


class FeatureEngineeringPipeline:
    def __init__(self, target=None, use_num=True, num_method="yeo-johnson",
                 use_cat=True, cat_method="onehot",
                 use_cluster=False, n_clusters=5,
                 use_arithmetic=False,
                 use_reduction=False, reduction_method="pca", n_components=5):
        self.target = target
        self.use_num = use_num
        self.num_method = num_method
        self.use_cat = use_cat
        self.cat_method = cat_method
        self.use_cluster = use_cluster
        self.n_clusters = n_clusters
        self.use_arithmetic = use_arithmetic
        self.use_reduction = use_reduction
        self.reduction_method = reduction_method
        self.n_components = n_components

        self.num_transformer = None
        self.cat_encoder = None
        self.clusterer = None
        self.arithmetic = None
        self.reducer = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        if self.use_num:
            num_cols = X.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                self.num_transformer = NumericalTransformer(self.num_method)
                self.num_transformer.fit(X[num_cols])

        if self.use_cat:
            cat_cols = X.select_dtypes(exclude=np.number).columns
            if len(cat_cols) > 0:
                self.cat_encoder = CategoricalEncoder(method=self.cat_method, target=self.target)
                self.cat_encoder.fit(X[cat_cols], y)

        if self.use_cluster:
            num_cols = X.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                self.clusterer = ClusterFeatures(self.n_clusters)
                self.clusterer.fit(X[num_cols])

        if self.use_arithmetic:
            self.arithmetic = ArithmeticFeatures()

        if self.use_reduction:
            self.reducer = FeatureReducer(method=self.reduction_method, n_components=self.n_components)
            self.reducer.fit(X.select_dtypes(include=np.number), y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = pd.DataFrame(index=X.index)

        if self.use_num:
            num_cols = X.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                X_num = self.num_transformer.transform(X[num_cols])
                X_new = pd.concat([X_new, X_num], axis=1)

        if self.use_cat:
            cat_cols = X.select_dtypes(exclude=np.number).columns
            if len(cat_cols) > 0:
                X_cat = self.cat_encoder.transform(X[cat_cols])
                X_new = pd.concat([X_new, X_cat], axis=1)

        if self.use_cluster:
            num_cols = X.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                X_cluster = self.clusterer.transform(X[num_cols])
                X_new = pd.concat([X_new, X_cluster], axis=1)

        if self.use_arithmetic:
            num_cols = X.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                X_arith = self.arithmetic.transform(X[num_cols])
                X_new = pd.concat([X_new, X_arith], axis=1)

        if self.use_reduction:
            X_reduced = self.reducer.transform(X.select_dtypes(include=np.number))
            X_new = pd.concat([X_new, X_reduced], axis=1)

        return X_new