"""
Models Package for Bank Customer Churn Prediction
=================================================

This package contains implementations of various machine learning models
following the 3rd place solution approach from Kaggle competition.

Available Models:
- XGBoost with Optuna optimization
- LightGBM with hyperparameter tuning
- CatBoost with different bootstrap types
- TensorFlow/Keras deep learning models
- Ensemble methods and model blending

Usage:
    from src.models import XGBoostChurnModel, LightGBMChurnModel, CatBoostEnsemble
    from src.models import EnsembleManager, WeightedEnsemble
"""

# Import base classes
from .base import (
    BaseChurnModel,
    CustomTransformers,
    FeatureDropper,
    Categorizer,
    Vectorizer,
    CAT_FEATURES,
    # Function transformers
    Nullify,
    SalaryRounder,
    AgeRounder,
    BalanceRounder,
    FeatureGenerator,
    SVDRounder
)

# Import individual models
from .xgboost import XGBoostChurnModel, XGBoostEnsemble
from .lightgbm import LightGBMChurnModel, LightGBMEnsemble
from .catboost import CatBoostChurnModel, CatBoostEnsemble, CatBoostAdvanced

# Import TensorFlow models (conditional)
try:
    from .tensorflow_model import (
        TensorFlower, 
        TensorFlowChurnModel, 
        TensorFlowEnsemble,
        AdvancedTensorFlower
    )
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Import ensemble methods
from .ensemble import (
    WeightedEnsemble,
    EnsembleManager,
    ModelBlender
)

# Define what's available when importing with *
__all__ = [
    # Base classes
    'BaseChurnModel',
    'CustomTransformers',
    'FeatureDropper',
    'Categorizer', 
    'Vectorizer',
    'CAT_FEATURES',
    
    # Function transformers
    'Nullify',
    'SalaryRounder',
    'AgeRounder', 
    'BalanceRounder',
    'FeatureGenerator',
    'SVDRounder',
    
    # Individual models
    'XGBoostChurnModel',
    'XGBoostEnsemble',
    'LightGBMChurnModel',
    'LightGBMEnsemble',
    'CatBoostChurnModel',
    'CatBoostEnsemble',
    'CatBoostAdvanced',
    
    # Ensemble methods
    'WeightedEnsemble',
    'EnsembleManager',
    'ModelBlender',
]

# Add TensorFlow models if available
if TENSORFLOW_AVAILABLE:
    __all__.extend([
        'TensorFlower',
        'TensorFlowChurnModel',
        'TensorFlowEnsemble', 
        'AdvancedTensorFlower'
    ])


def get_available_models():
    """
    Get list of available models.
    
    Returns:
        dict: Dictionary of available models and their status
    """
    models_status = {
        'XGBoost': True,
        'LightGBM': True, 
        'CatBoost': True,
        'TensorFlow': TENSORFLOW_AVAILABLE,
        'Ensemble Methods': True
    }
    
    return models_status


def create_model_pipeline():
    """
    Create a complete model training pipeline following the 3rd place solution.
    
    Returns:
        EnsembleManager: Configured ensemble manager
    """
    # Initialize ensemble manager
    manager = EnsembleManager(seed=42, n_splits=30)
    
    print("🏗️  Creating model pipeline following 3rd place solution...")
    print("📋 Available models:")
    
    for model_name, available in get_available_models().items():
        status = "✅" if available else "❌"
        print(f"   {status} {model_name}")
    
    return manager


def print_package_info():
    """Print package information and available models."""
    print("=" * 60)
    print("🏆 Bank Customer Churn Prediction - 3rd Place Solution")
    print("=" * 60)
    print("\n📦 Models Package Information:")
    print(f"   Version: 1.0.0")
    print(f"   Models Available: {len([k for k, v in get_available_models().items() if v])}")
    
    print("\n🤖 Available Models:")
    for model_name, available in get_available_models().items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"   • {model_name}: {status}")
    
    if not TENSORFLOW_AVAILABLE:
        print("\n💡 To enable TensorFlow models:")
        print("   pip install tensorflow")
    
    print("\n🚀 Quick Start:")
    print("   from src.models import create_model_pipeline")
    print("   manager = create_model_pipeline()")
    print("=" * 60)


if __name__ == "__main__":
    print_package_info()
