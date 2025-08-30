"""
TensorFlow/Keras Model Implementation for Bank Customer Churn Prediction
=======================================================================

This module implements deep learning models using TensorFlow/Keras
following the 3rd place solution approach.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from category_encoders import CatBoostEncoder

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    # Set deterministic operations for reproducibility
    tf.keras.utils.set_random_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available. Install with: pip install tensorflow")

from .base import (
    BaseChurnModel, SalaryRounder, AgeRounder, FeatureGenerator, 
    Vectorizer, CAT_FEATURES
)


class TensorFlower(BaseEstimator, ClassifierMixin):
    """
    Custom TensorFlow/Keras classifier following the 3rd place solution architecture.
    """
    
    def __init__(self, seed=42):
        """
        Initialize TensorFlow model.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed
        self.model = None
        self.classes_ = None
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for this model. Install with: pip install tensorflow")
    
    def _build_model(self, input_shape):
        """
        Build the neural network architecture from the 3rd place solution.
        
        Args:
            input_shape (int): Number of input features
            
        Returns:
            tf.keras.Model: Compiled model
        """
        inputs = tf.keras.Input((input_shape,))
        inputs_norm = tf.keras.layers.BatchNormalization()(inputs)
        
        # First hidden layer
        z = tf.keras.layers.Dense(32)(inputs_norm)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.LeakyReLU()(z)
        
        # Second hidden layer
        z = tf.keras.layers.Dense(64)(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.LeakyReLU()(z)
        
        # Third hidden layer
        z = tf.keras.layers.Dense(16)(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.LeakyReLU()(z)
        
        # Fourth hidden layer
        z = tf.keras.layers.Dense(4)(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.LeakyReLU()(z)
        
        # Output layer
        z = tf.keras.layers.Dense(1)(z)
        z = tf.keras.layers.BatchNormalization()(z)
        outputs = tf.keras.activations.sigmoid(z)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            loss='binary_crossentropy', 
            optimizer=tf.keras.optimizers.AdamW(1e-4)
        )
        
        return model
    
    def fit(self, x, y):
        """
        Fit the TensorFlow model.
        
        Args:
            x: Input features
            y: Target labels
            
        Returns:
            self
        """
        self.classes_ = np.unique(y)
        
        # Build model
        self.model = self._build_model(x.shape[1])
        
        # Train model
        self.model.fit(
            x.to_numpy() if hasattr(x, 'to_numpy') else x, 
            y, 
            epochs=10, 
            verbose=0
        )
        
        return self
    
    def predict_proba(self, x):
        """
        Predict class probabilities.
        
        Args:
            x: Input features
            
        Returns:
            np.ndarray: Class probabilities
        """
        predictions = np.zeros((len(x), 2))
        predictions[:, 1] = self.model.predict(x, verbose=0)[:, 0]
        predictions[:, 0] = 1 - predictions[:, 1]
        return predictions
    
    def predict(self, x):
        """
        Predict class labels.
        
        Args:
            x: Input features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        return np.argmax(self.predict_proba(x), axis=1)


class TensorFlowChurnModel(BaseChurnModel):
    """
    TensorFlow implementation for churn prediction.
    """
    
    def __init__(self, seed=42, n_splits=30):
        """
        Initialize TensorFlow model.
        
        Args:
            seed (int): Random seed
            n_splits (int): Number of CV folds
        """
        super().__init__(seed, n_splits)
        self.model = None
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for this model. Install with: pip install tensorflow")
    
    def create_pipeline(self):
        """
        Create TensorFlow pipeline following the 3rd place solution.
        
        Returns:
            sklearn.pipeline.Pipeline: Complete preprocessing + model pipeline
        """
        pipeline = make_pipeline(
            SalaryRounder,
            AgeRounder,
            FeatureGenerator,
            # Note: Vectorizer commented out in original solution for TF model
            # Vectorizer(cols=['Surname', 'AllCat', 'EstimatedSalary', 'CreditScore'], 
            #           max_features=500, n_components=6),
            CatBoostEncoder(cols=CAT_FEATURES),
            TensorFlower(seed=self.seed)
        )
        
        return pipeline
    
    def train_and_predict(self, train_df, orig_train_df, test_df, 
                         show_importance=False, label="TensorFlow"):
        """
        Train model and make predictions.
        
        Args:
            train_df: Training data
            orig_train_df: Original dataset
            test_df: Test data
            show_importance: Whether to show feature importance
            label: Model label for display
            
        Returns:
            Tuple of (validation_scores, oof_predictions, test_predictions)
        """
        # Create model
        self.model = self.create_pipeline()
        
        print(f"🚀 Training {label} model...")
        val_scores, oof_preds, test_preds = self.cross_val_score(
            self.model, train_df, orig_train_df, test_df,
            label=label, show_importance=show_importance
        )
        
        return val_scores, oof_preds, test_preds


class TensorFlowEnsemble:
    """
    Ensemble of different TensorFlow configurations.
    """
    
    def __init__(self, seed=42, n_splits=30):
        self.seed = seed
        self.n_splits = n_splits
        self.models = {}
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for this ensemble. Install with: pip install tensorflow")
        
    def create_variant_models(self):
        """Create different TensorFlow model variants."""
        
        # Standard TensorFlow model
        self.models['tf_standard'] = TensorFlowChurnModel(
            seed=self.seed, n_splits=self.n_splits
        )
        
        # Additional variants could be added here with different architectures
        
        return self.models
    
    def train_ensemble(self, train_df, orig_train_df, test_df):
        """
        Train ensemble of TensorFlow models.
        
        Args:
            train_df: Training data
            orig_train_df: Original dataset
            test_df: Test data
            
        Returns:
            dict: Results from all models
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}")
            print(f"{'='*50}")
            
            val_scores, oof_preds, test_preds = model.train_and_predict(
                train_df, orig_train_df, test_df, label=name
            )
            
            results[name] = {
                'val_scores': val_scores,
                'oof_predictions': oof_preds,
                'test_predictions': test_preds,
                'mean_score': np.mean(val_scores),
                'std_score': np.std(val_scores)
            }
            
        return results


class AdvancedTensorFlower(BaseEstimator, ClassifierMixin):
    """
    Advanced TensorFlow model with more sophisticated architecture.
    """
    
    def __init__(self, seed=42, architecture='deep', dropout_rate=0.3, 
                 learning_rate=1e-4, epochs=50):
        """
        Initialize advanced TensorFlow model.
        
        Args:
            seed (int): Random seed
            architecture (str): Model architecture ('deep', 'wide', 'wide_deep')
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            epochs (int): Number of training epochs
        """
        self.seed = seed
        self.architecture = architecture
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.classes_ = None
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for this model. Install with: pip install tensorflow")
    
    def _build_deep_model(self, input_shape):
        """Build deep neural network architecture."""
        inputs = tf.keras.Input((input_shape,))
        
        # Deep component
        x = tf.keras.layers.BatchNormalization()(inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs, outputs)
        return model
    
    def _build_model(self, input_shape):
        """Build model based on architecture choice."""
        if self.architecture == 'deep':
            model = self._build_deep_model(input_shape)
        else:
            # Default to original architecture
            model = TensorFlower(self.seed)._build_model(input_shape)
        
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.AdamW(self.learning_rate),
            metrics=['AUC']
        )
        
        return model
    
    def fit(self, x, y):
        """Fit the advanced model."""
        self.classes_ = np.unique(y)
        self.model = self._build_model(x.shape[1])
        
        # Add callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.5, patience=5, min_lr=1e-7
            )
        ]
        
        self.model.fit(
            x.to_numpy() if hasattr(x, 'to_numpy') else x,
            y,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=0
        )
        
        return self
    
    def predict_proba(self, x):
        """Predict probabilities."""
        predictions = np.zeros((len(x), 2))
        predictions[:, 1] = self.model.predict(x, verbose=0)[:, 0]
        predictions[:, 0] = 1 - predictions[:, 1]
        return predictions
    
    def predict(self, x):
        """Predict class labels."""
        return np.argmax(self.predict_proba(x), axis=1)


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing TensorFlow implementations...")
    
    if TF_AVAILABLE:
        # Test standard model
        model = TensorFlowChurnModel()
        print("✅ Standard TensorFlow model ready!")
        
        # Test advanced model
        advanced_model = AdvancedTensorFlower()
        print("✅ Advanced TensorFlow model ready!")
        
        # Test ensemble
        ensemble = TensorFlowEnsemble()
        ensemble.create_variant_models()
        print(f"✅ TensorFlow ensemble ready!")
    else:
        print("❌ TensorFlow not available. Install with: pip install tensorflow")
    
    print("✅ TensorFlow module ready for use!")
