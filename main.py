# main.py
import yaml
import pandas as pd
import logging
from pathlib import Path

# Custom module imports
from src.modules.data_processing import DataLoader, DataSplitter
from src.modules.feature_engineering import FeatureEngineeringPipeline
from src.modules.imbalance_handler import ImbalanceHandler
from src.models.evaluation import get_evaluation_metrics, print_classification_report
from src.models.xgboost import XGBoostModel
from src.models.lightgbm import LightGBMModel
from src.models.catboost import CatBoostModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path='config.yaml'):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_model_instance(model_name: str, params: dict):
    """Factory function to get a model instance."""
    model_map = {
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'catboost': CatBoostModel
    }
    model_class = model_map.get(model_name)
    if not model_class:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return model_class(params=params)

def run_pipeline():
    """Main function to execute the full training and evaluation pipeline."""
    
    # 1. Load Configuration
    config = load_config()
    cfg_dp = config['data_processing']
    cfg_fe = config['feature_engineering']
    cfg_imb = config['imbalance_handling']
    logging.info("Configuration loaded successfully.")

    # 2. Load and Combine Datasets
    data_loader = DataLoader()
    full_train_df = data_loader.combine_datasets()
    logging.info(f"Data loaded and combined. Shape: {full_train_df.shape}")

    # 3. Split Data into Training and Test Sets (Crucial First Step)
    data_splitter = DataSplitter(
        test_size=cfg_dp['test_split_size'],
        random_state=cfg_dp['random_state']
    )
    X_train, X_test, y_train, y_test = data_splitter.split(full_train_df, target_col=cfg_dp['target_column'])
    logging.info(f"Data split. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Initialize and Fit the Feature Engineering Pipeline on Training Data
    feature_pipeline = FeatureEngineeringPipeline(
        use_num=cfg_fe['use_num'],
        num_method=cfg_fe['num_transform_method'],
        use_cat=cfg_fe['use_cat'],
        cat_method=cfg_fe['cat_encoding_method'],
        use_cluster=cfg_fe['use_cluster'],
        n_clusters=cfg_fe['n_clusters'],
        use_arithmetic=cfg_fe['use_arithmetic'],
        use_reduction=cfg_fe['use_reduction'],
        reduction_method=cfg_fe['reduction_method'],
        n_components=cfg_fe['n_components']
    )
    logging.info("Fitting feature engineering pipeline on training data...")
    feature_pipeline.fit(X_train, y_train)
    
    # Transform the training data
    X_train_processed = feature_pipeline.transform(X_train)
    logging.info(f"Training data transformed. New shape: {X_train_processed.shape}")

    # 5. Transform the Test Data using the Fitted Pipeline (do this once)
    logging.info("Transforming test data with the fitted pipeline...")
    X_test_final = feature_pipeline.transform(X_test)

    # 6. Train Models with Different Imbalance Handling Methods
    all_results = []
    saved_model_dir = Path(config['paths']['saved_model_dir'])
    saved_model_dir.mkdir(exist_ok=True) # Create directory if it doesn't exist

    # Loop through each imbalance handling method
    for imbalance_method in cfg_imb['methods']:
        logging.info(f"\n{'='*50}")
        logging.info(f"IMBALANCE HANDLING METHOD: {imbalance_method.upper()}")
        logging.info(f"{'='*50}")
        
        # Apply imbalance handling
        if imbalance_method.lower() != 'none':
            imbalancer = ImbalanceHandler(method=imbalance_method, random_state=cfg_imb['random_state'])
            logging.info(f"Applying '{imbalance_method}' to handle class imbalance...")
            X_train_final, y_train_final = imbalancer.fit_resample(X_train_processed, y_train)
            logging.info(f"Original training shape: {X_train_processed.shape}")
            logging.info(f"Resampled training shape: {X_train_final.shape}")
            
            # Check class distribution
            class_counts = pd.Series(y_train_final).value_counts().sort_index()
            logging.info(f"Class distribution after {imbalance_method}: {dict(class_counts)}")
        else:
            X_train_final, y_train_final = X_train_processed, y_train
            logging.info("No imbalance handling applied.")
            class_counts = pd.Series(y_train_final).value_counts().sort_index()
            logging.info(f"Original class distribution: {dict(class_counts)}")

        # Train each model with current imbalance method
        for model_name, model_config in config['models'].items():
            if not model_config.get('train', False):
                continue

            logging.info(f"\n{'-'*20} Training {model_name.upper()} with {imbalance_method.upper()} {'-'*20}")
            
            # Instantiate and train model
            model = get_model_instance(model_name, model_config['params'])
            
            # Convert DataFrame to numpy array for model training
            X_train_array = X_train_final.values if hasattr(X_train_final, 'values') else X_train_final
            y_train_array = y_train_final.values if hasattr(y_train_final, 'values') else y_train_final
            
            model.fit(X_train_array, y_train_array)
            
            # Make predictions on the transformed test set
            X_test_array = X_test_final.values if hasattr(X_test_final, 'values') else X_test_final
            y_pred = model.predict(X_test_array)
            y_pred_proba = model.predict_proba(X_test_array)[:, 1]
            
            # Evaluate
            model_identifier = f"{model_name}_{imbalance_method}"
            metrics = get_evaluation_metrics(y_test, y_pred, y_pred_proba, model_identifier)
            all_results.append(metrics)
            
            logging.info(f"Evaluation Report for {model_name.upper()} with {imbalance_method.upper()}:")
            print_classification_report(y_test, y_pred)

            # Save model with imbalance method in name
            model_save_path = saved_model_dir / f"{model_name}_{imbalance_method}_model.joblib"
            model.save(model_save_path)
            logging.info(f"{model_name.upper()}_{imbalance_method.upper()} model saved to {model_save_path}")

    # 7. Display Final Comparative Results
    results_df = pd.DataFrame(all_results).set_index('Model')
    logging.info(f"\n{'='*25} FINAL COMPARATIVE ANALYSIS SUMMARY {'='*25}")
    print(results_df)
    
    # Display results grouped by model for easier comparison
    logging.info(f"\n{'='*25} RESULTS BY MODEL {'='*25}")
    for model_name in config['models'].keys():
        if config['models'][model_name].get('train', False):
            model_results = results_df[results_df.index.str.startswith(model_name)]
            if not model_results.empty:
                logging.info(f"\n{model_name.upper()} Results:")
                print(model_results)
    
    logging.info("Pipeline execution finished successfully! 🎉")

if __name__ == "__main__":
    run_pipeline()