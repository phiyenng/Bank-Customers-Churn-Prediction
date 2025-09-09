# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Bank Customer Churn Prediction is a machine learning pipeline for predicting customer churn using multiple gradient boosting models (XGBoost, LightGBM, CatBoost) with various class imbalance handling techniques.

## Key Commands

### Development & Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Run individual notebooks for analysis
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_modeling.ipynb
jupyter notebook notebooks/04_evaluation.ipynb
```

### Configuration
- Edit `config.yaml` to modify data processing, feature engineering, imbalance handling, and model parameters
- Individual model configs available in `configs/` directory

## Architecture & Code Structure

### Core Pipeline Flow (main.py)
1. **Data Loading** (`DataLoader`) - Combines original and competition datasets
2. **Data Splitting** (`DataSplitter`) - Splits into train/test before any processing
3. **Feature Engineering** (`FeatureEngineeringPipeline`) - Fitted on training data only
4. **Imbalance Handling** (`ImbalanceHandler`) - Applied per method iteration
5. **Model Training** - Multiple models with different imbalance techniques
6. **Evaluation** - Comprehensive metrics comparison

### Key Design Principles
- **Data Leakage Prevention**: Test data is split before any processing and transformed only with fitted pipelines
- **Modular Architecture**: Each component (data processing, feature engineering, models) is separate and configurable
- **Factory Pattern**: Model instantiation through `get_model_instance()` factory function
- **Configuration-Driven**: All parameters controlled via YAML config files

### Module Structure
```
src/
├── modules/
│   ├── data_processing.py      # DataLoader, DataSplitter classes
│   ├── feature_engineering.py  # Comprehensive FE pipeline with transformers
│   └── imbalance_handler.py     # SMOTE, undersampling, hybrid methods
└── models/
    ├── base_model.py           # Abstract BaseModel class
    ├── evaluation.py           # Metrics calculation and reporting
    ├── xgboost.py             # XGBoost wrapper
    ├── lightgbm.py            # LightGBM wrapper
    └── catboost.py            # CatBoost wrapper
```

### Feature Engineering Pipeline Components
- **Numerical Transformers**: Yeo-Johnson, Box-Cox, log, sqrt transformations
- **Categorical Encoders**: OneHot, Label, Target encoding
- **Cluster Features**: K-means clustering for feature enrichment
- **Arithmetic Features**: Ratio, product, difference calculations
- **Dimensionality Reduction**: PCA, SVD, Chi2 feature selection

### Model Training Strategy
The pipeline trains each model (XGBoost, LightGBM, CatBoost) with multiple imbalance handling methods:
- None (baseline)
- SMOTE
- SMOTEENN (hybrid)
- NearMiss (undersampling)

Models are saved with naming pattern: `{model_name}_{imbalance_method}_model.joblib`

### Configuration Management
- **Main Config**: `config.yaml` - Controls entire pipeline
- **Model-Specific**: `configs/{model}_config.yaml` - Individual model parameters
- **Paths**: All file paths centrally managed in config
- **Reproducibility**: Fixed random seeds throughout

### Data Flow & Artifacts
```
data/raw/ → DataLoader → DataSplitter → FeatureEngineeringPipeline → ImbalanceHandler → Models → saved_models/
                                     ↓
                              artifacts/ (fitted pipelines, encoders, scalers)
```

### Critical Implementation Notes
- **Feature Pipeline Fitting**: Always fit on training data before any resampling
- **Test Data Transformation**: Use fitted pipeline, never refit on test data
- **Model Evaluation**: Each model tested on same transformed test set for fair comparison
- **Memory Management**: DataFrames converted to numpy arrays for model training
- **Error Handling**: FileNotFoundError handling for missing datasets

## Development Guidelines

### Adding New Models
1. Inherit from `BaseModel` in `src/models/base_model.py`
2. Implement `_create_model()` method
3. Add to model factory in `main.py`
4. Update config.yaml with model parameters

### Adding Imbalance Methods
1. Add method to `ImbalanceHandler` class
2. Update `_init_sampler()` method
3. Add to config.yaml imbalance_handling methods list

### Feature Engineering Extensions
1. Create new transformer class following existing patterns
2. Add to `FeatureEngineeringPipeline`
3. Update config.yaml with new parameters

### Testing New Configurations
- Reduce `n_estimators` in config for faster iterations
- Set `train: false` for models you want to skip
- Use smaller `n_clusters` and `n_components` for quicker testing
