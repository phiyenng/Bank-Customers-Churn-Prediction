# Models Package - 3rd Place Solution Implementation

## 📋 Overview

This package contains a complete implementation of machine learning models following the **3rd place solution** approach from the Kaggle "Binary Classification with a Bank Churn Dataset" competition.

## 🏗️ Architecture

The implementation follows a modular architecture with the following components:

```
src/models/
├── base.py              # Base classes and common transformers
├── xgboost.py          # XGBoost implementation with Optuna
├── lightgbm.py         # LightGBM with hyperparameter tuning
├── catboost.py         # CatBoost with different bootstrap types
├── tensorflow_model.py  # Deep learning models
├── ensemble.py         # Ensemble methods and model blending
└── __init__.py         # Package initialization
```

## 🚀 Quick Start

### Basic Usage

```python
# Import the training pipeline
from train_models import ChurnPredictionPipeline

# Initialize pipeline
pipeline = ChurnPredictionPipeline(seed=42, n_splits=30, use_optuna=False)

# Run complete training
pipeline.run_complete_pipeline()
```

### Individual Models

```python
from src.models import XGBoostChurnModel, LightGBMChurnModel, CatBoostEnsemble

# Train XGBoost
xgb_model = XGBoostChurnModel(use_optuna=False)
val_scores, oof_preds, test_preds = xgb_model.train_and_predict(train_df, orig_train_df, test_df)

# Train CatBoost ensemble
catboost_ensemble = CatBoostEnsemble()
catboost_ensemble.create_variant_models()
results = catboost_ensemble.train_ensemble(train_df, orig_train_df, test_df)
```

### Ensemble Methods

```python
from src.models import EnsembleManager, WeightedEnsemble

# Create ensemble manager
manager = EnsembleManager()

# Add model results
manager.add_model_results("XGB", val_scores, oof_preds, test_preds)
manager.add_model_results("LGB", val_scores, oof_preds, test_preds)

# Create weighted ensemble
ensemble = manager.create_weighted_ensemble(target)
```

## 🤖 Available Models

### 1. XGBoost (`xgboost.py`)
- **XGBoostChurnModel**: Standard XGBoost with optimized parameters
- **XGBoostEnsemble**: Multiple XGBoost configurations
- **Features**: 
  - Optuna hyperparameter optimization
  - Custom feature engineering pipeline
  - Cross-validation with original dataset inclusion

### 2. LightGBM (`lightgbm.py`)
- **LightGBMChurnModel**: LightGBM with tuned parameters
- **LightGBMEnsemble**: Multiple LightGBM variants including DART
- **Features**:
  - Hyperparameter optimization
  - Different boosting types
  - Advanced regularization

### 3. CatBoost (`catboost.py`)
- **CatBoostChurnModel**: Base CatBoost implementation
- **CatBoostEnsemble**: Multiple bootstrap types (Ordered, Bayesian, Bernoulli)
- **CatBoostAdvanced**: Enhanced version with additional features
- **Features**:
  - Native categorical feature handling
  - Time-based ordering (`has_time=True`)
  - Multiple bootstrap strategies

### 4. TensorFlow (`tensorflow_model.py`)
- **TensorFlower**: Custom Keras classifier
- **TensorFlowChurnModel**: Deep learning pipeline
- **AdvancedTensorFlower**: Enhanced architecture
- **Features**:
  - Batch normalization layers
  - LeakyReLU activations
  - AdamW optimizer

### 5. Ensemble Methods (`ensemble.py`)
- **WeightedEnsemble**: Ridge regression-based weighting
- **EnsembleManager**: Complete ensemble workflow
- **ModelBlender**: Advanced blending strategies
- **Features**:
  - Rank averaging
  - Geometric mean blending
  - Power averaging

## 🔧 Key Features from 3rd Place Solution

### Data Strategy
- **Dataset Combination**: Original dataset placed first for stable encoding
- **Data Leakage Prevention**: Specific handling of known leakage issues
- **Cross-Validation**: 30-fold StratifiedKFold for robust validation

### Feature Engineering
- **Custom Transformers**: Salary/Age/Balance rounding for better encoding
- **Advanced Features**: Product tenure ratios, activity combinations
- **Text Vectorization**: TF-IDF + SVD for categorical features
- **Categorical Encoding**: CatBoost and MEstimate encoders

### Model-Specific Optimizations
- **XGBoost**: Optimized tree parameters with histogram method
- **LightGBM**: Advanced regularization and DART boosting
- **CatBoost**: Multiple bootstrap types with categorical features
- **TensorFlow**: Batch normalization and LeakyReLU architecture

## 📊 Performance Expectations

Based on the 3rd place solution:
- **Individual Models**: ~0.88-0.90 ROC AUC
- **Ensemble**: ~0.90+ ROC AUC
- **Cross-Validation**: 30-fold for robust estimates

## ⚙️ Configuration

### Default Parameters
```python
SEED = 42
N_SPLITS = 30
USE_OPTUNA = False  # Set True for hyperparameter optimization
```

### Hyperparameter Optimization
```python
# Enable Optuna optimization (slower but better results)
xgb_model = XGBoostChurnModel(use_optuna=True, n_trials=50)
lgb_model = LightGBMChurnModel(use_optuna=True, n_trials=100)
```

## 📁 Input Data Requirements

The models expect data in the following format:

```
data/raw/
├── Churn_Modelling.csv     # Original dataset
├── train.csv               # Competition training data  
└── test.csv                # Competition test data
```

### Required Columns
- `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`
- `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`
- `EstimatedSalary`, `Exited` (target, for training data)

## 🎯 Expected Output

### Submission File
- `submission.csv`: Predictions in competition format
- Data leakage corrections applied automatically
- Probability scores between 0 and 1

### Model Artifacts
- Cross-validation scores for each model
- Feature importance rankings
- Ensemble weights
- Performance visualizations

## 🔍 Advanced Usage

### Custom Feature Engineering
```python
from src.models.base import FeatureGenerator, Vectorizer

# Custom vectorization
vectorizer = Vectorizer(
    cols=['Surname', 'AllCat'], 
    max_features=1000, 
    n_components=4
)

# Add to pipeline
pipeline = make_pipeline(
    FeatureGenerator,
    vectorizer,
    CatBoostClassifier()
)
```

### Custom Ensemble Weights
```python
# Manual weight assignment
custom_weights = {
    'XGB': 0.3,
    'LGB': 0.25, 
    'CB_Bayes': 0.25,
    'CB_Bernoulli': 0.2
}

# Apply weights
weighted_preds = sum(preds[model] * weight 
                    for model, weight in custom_weights.items())
```

## 🚨 Important Notes

1. **Data Leakage Handling**: The solution includes specific fixes for known data leakage in the competition
2. **Memory Usage**: Some models (especially with vectorization) can be memory-intensive
3. **Training Time**: Full pipeline with 30-fold CV takes significant time
4. **Reproducibility**: All models use fixed seeds for consistent results

## 🔧 Troubleshooting

### Common Issues
1. **Memory Error**: Reduce `max_features` in Vectorizer or use fewer CV folds
2. **Optuna Timeout**: Reduce `n_trials` in optimization
3. **TensorFlow Missing**: Install with `pip install tensorflow`
4. **Slow Training**: Reduce `n_splits` or disable Optuna

### Performance Tips
- Use `USE_OPTUNA=False` for faster training
- Reduce `n_splits` from 30 to 5-10 for testing
- Enable GPU for TensorFlow models if available

## 📈 Expected Competition Score

Following the exact 3rd place solution approach should yield:
- **Public Leaderboard**: ~0.90 ROC AUC
- **Private Leaderboard**: ~0.90+ ROC AUC (3rd place performance)

## 🏆 Credits

This implementation is based on the 3rd place solution from the Kaggle competition "Binary Classification with a Bank Churn Dataset". All credit goes to the original solution authors for their innovative approach to dataset combination, feature engineering, and ensemble methods.
