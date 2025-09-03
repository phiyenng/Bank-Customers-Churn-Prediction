import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from models.utils import cross_val_score
from modules.feature_engineering import FeatureEngineeringPipeline

class TensorFlower(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        inputs = tf.keras.Input(shape=(X.shape[1],))
        z = tf.keras.layers.BatchNormalization()(inputs)
        for units in [32, 64, 16, 4]:
            z = tf.keras.layers.Dense(units)(z)
            z = tf.keras.layers.BatchNormalization()(z)
            z = tf.keras.layers.LeakyReLU()(z)
        z = tf.keras.layers.Dense(1)(z)
        z = tf.keras.layers.BatchNormalization()(z)
        outputs = tf.keras.activations.sigmoid(z)

        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(loss="binary_crossentropy",
                           optimizer=tf.keras.optimizers.AdamW(1e-4))
        self.model.fit(X.to_numpy(), y, epochs=10, verbose=0)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        preds = np.zeros((len(X), 2))
        preds[:, 1] = self.model.predict(X, verbose=0)[:, 0]
        preds[:, 0] = 1 - preds[:, 1]
        return preds

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

def get_tensorflow_pipeline():
    return make_pipeline(
        FeatureEngineeringPipeline(),
        TensorFlower()
    )
