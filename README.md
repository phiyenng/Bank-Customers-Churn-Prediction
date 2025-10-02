# Bank Customer Churn Prediction

A machine learning pipeline for predicting bank customer churn using multiple algorithms and imbalance handling techniques. This project implements a comprehensive comparison of different models (XGBoost, LightGBM, CatBoost) with various sampling strategies to identify customers likely to leave the bank.

## Project Structure

```
Bank-Customers-Churn-Prediction/
├── data/
│   └── raw/
│       ├── Churn_Modelling.csv    # Main dataset
│       ├── train.csv              # Training data
│       └── test.csv               # Test data
├── src/
│   ├── models/
│   │   ├── xgboost.py            # XGBoost model implementation
│   │   ├── lightgbm.py           # LightGBM model implementation
│   │   ├── catboost.py           # CatBoost model implementation
│   │   ├── train_cv.py           # Cross-validation training
│   │   └── evaluation.py         # Model evaluation utilities
│   └── modules/
│       ├── processing.py         # Data loading and preprocessing
│       ├── feature_engineering.py # Feature creation and selection
│       └── imbalance_handler.py  # Imbalance handling methods
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_preprocessing.ipynb    # Data preprocessing
│   └── 03_evaluation.ipynb      # Model evaluation
├── *_saved_models/              # Model artifacts for different methods
│   ├── best_model_tuned_optuna.joblib
│   ├── cv/                      # Cross-validation results
│   ├── plots/                   # Evaluation plots
│   └── test_metrics_*.csv       # Performance metrics
├── config.yaml                 # Configuration file
├── main.py                     # Main training pipeline
├── app.py                      # Streamlit web application
├── run_app.py                  # App launcher
└── requirements.txt            # Python dependencies
```

## Model Results

The project evaluates multiple imbalance handling techniques with LightGBM as the primary model. Here are the best results from hyperparameter tuning:

| Method | Accuracy | F1 Score | Precision | Recall | ROC-AUC | PR-AUC |
|--------|----------|----------|-----------|--------|---------|--------|
| ADASYN | 0.865 | 0.861 | 0.859 | 0.865 | 0.884 | 0.701 |
| SMOTE | 0.865 | 0.861 | 0.859 | 0.865 | 0.884 | 0.701 |
| Class Weight | 0.809 | 0.823 | 0.856 | 0.809 | 0.885 | 0.702 |
| SMOTE-ENN | 0.865 | 0.861 | 0.859 | 0.865 | 0.884 | 0.701 |
| SMOTE-Tomek | 0.865 | 0.861 | 0.859 | 0.865 | 0.884 | 0.701 |

**Best Performance**: ADASYN, SMOTE, SMOTE-ENN, and SMOTE-Tomek methods achieve similar high performance with ~86.5% accuracy and 0.884 ROC-AUC.

## How to Run

### Prerequisites

1. Install Python 3.8+ and create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Model Training

1. **Basic training with default configuration:**
```bash
python main.py
```

2. **Custom configuration:**
   - Edit `config.yaml` to modify:
     - Imbalance handling method (`smote`, `adasyn`, `class_weight`, etc.)
     - Model parameters and hyperparameter tuning settings
     - Cross-validation and evaluation settings
   - Run the pipeline:
```bash
python main.py
```

The training pipeline will:
- Load and preprocess the data
- Apply feature engineering and selection
- Handle class imbalance using the specified method
- Train models with cross-validation
- Perform hyperparameter tuning with Optuna
- Generate evaluation plots and SHAP explanations
- Save the best model and results

### Running Streamlit Web Application

1. **Launch the web app:**
```bash
streamlit run app.py
```
   Or use the launcher:
```bash
python run_app.py
```

2. **Access the application:**
   - Open your browser to `http://localhost:8501`
   - The app provides:
     - Interactive customer churn prediction
     - Model performance visualization
     - Feature importance analysis
     - Data exploration tools

### Configuration Options

Key settings in `config.yaml`:

- **Imbalance handling**: Choose from `smote`, `adasyn`, `smote_tomek`, `smote_enn`, or `class_weight`
- **Models**: Enable/disable XGBoost, LightGBM, CatBoost
- **Hyperparameter tuning**: Adjust number of trials and parameter ranges
- **Feature selection**: Configure correlation threshold and selection methods
- **Cross-validation**: Set number of folds and validation strategy

### Output Files

After training, check these directories:
- `{method}_saved_models/`: Contains trained models and evaluation results
- `{method}_saved_models/plots/`: ROC curves, confusion matrices, SHAP plots
- `{method}_saved_models/cv/`: Cross-validation fold data and metrics

## Requirements

- Python 3.8+
- scikit-learn, pandas, numpy
- XGBoost, LightGBM, CatBoost
- Streamlit, Plotly (for web app)
- Optuna (for hyperparameter tuning)
- SHAP (for model interpretability)

See `requirements.txt` for complete dependency list.