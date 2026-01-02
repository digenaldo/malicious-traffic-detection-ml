import logging
from pathlib import Path
from src.utils.data_preprocessing import load_data, check_and_replace_invalid_values
from src.utils.feature_engineering import get_feature_engineering
from src.utils.model_training import create_pipeline, tune_hyperparameters as tune_hyperparams
from src.utils.evaluation import evaluate_model
from src.config import DATA_DIR, RESULTS_DIR, MODELS, FEATURE_ENGINEERING

def run_pipeline(algorithm, feature_engineering, tune_hyperparameters, cross_validation):
    """
    Run the complete machine learning pipeline for malicious traffic detection.
    
    Args:
        algorithm: Name of the ML algorithm to use
        feature_engineering: Name of the feature engineering technique
        tune_hyperparameters: Whether to perform hyperparameter tuning
        cross_validation: Whether to use cross-validation for evaluation
    """
    try:
        # Create results directory if it doesn't exist
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        train_data, test_data = load_data(DATA_DIR)
        check_and_replace_invalid_values(train_data, "train_data")
        check_and_replace_invalid_values(test_data, "test_data")
        
        # Separate features and target
        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]
        
        logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        # Get model and feature engineering technique
        if algorithm not in MODELS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        model, param_grid = MODELS[algorithm]
        feature_engineering_transformer = get_feature_engineering(feature_engineering)
        
        # Create pipeline
        pipeline = create_pipeline(model, feature_engineering_transformer)
        
        # Tune hyperparameters if requested
        if tune_hyperparameters:
            logging.info("Starting hyperparameter tuning...")
            pipeline, best_params = tune_hyperparams(pipeline, param_grid, X_train, y_train)
            logging.info(f"Best hyperparameters: {best_params}")
        else:
            logging.info("Training model without hyperparameter tuning...")
            pipeline.fit(X_train, y_train)
            best_params = None
        
        # Evaluate model
        accuracy, cv_mean, training_time = evaluate_model(
            pipeline,
            X_test,
            y_test,
            algorithm,
            feature_engineering,
            RESULTS_DIR,
            tune=tune_hyperparameters,
            cross_validation=cross_validation,
            X_train=X_train,
            y_train=y_train
        )

        logging.info(f"Final model accuracy: {accuracy:.4f}")
        logging.info(f"Cross-validation mean accuracy: {cv_mean:.4f}")
        logging.info(f"Training time: {training_time:.2f} seconds")
        
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error in pipeline execution: {str(e)}", exc_info=True)
        raise
