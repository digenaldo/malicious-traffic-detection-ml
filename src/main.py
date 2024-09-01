import logging
from src.utils.data_preprocessing import load_data, check_and_replace_invalid_values
from src.utils.feature_engineering import get_feature_engineering
from src.utils.model_training import create_pipeline, tune_hyperparameters
from src.utils.evaluation import evaluate_model
from src.config import DATA_DIR, RESULTS_DIR, MODELS, FEATURE_ENGINEERING

def run_pipeline(algorithm, feature_engineering, tune_hyperparameters, cross_validation):
    # Load and preprocess data
    train_data, test_data = load_data(DATA_DIR)
    check_and_replace_invalid_values(train_data, "train_data")
    check_and_replace_invalid_values(test_data, "test_data")
    
    # Separate features and target
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    
    # Get model and feature engineering technique
    model, param_grid = MODELS[algorithm]
    feature_engineering_transformer = get_feature_engineering(feature_engineering)
    
    # Create pipeline
    pipeline = create_pipeline(model, feature_engineering_transformer)
    
    # Tune hyperparameters if requested
    if tune_hyperparameters:
        pipeline, best_params = tune_hyperparameters(pipeline, param_grid, X_train, y_train)
    else:
        pipeline.fit(X_train, y_train)
        best_params = None
    
    # Evaluate model
    accuracy, cv_mean, training_time = evaluate_model(pipeline, X_test, y_test, algorithm, feature_engineering, RESULTS_DIR, tune=tune_hyperparameters)

    logging.info(f"Final model accuracy: {accuracy:.4f}")
    logging.info(f"Final cross-validation mean accuracy: {cv_mean:.4f}")
    logging.info(f"Training time: {training_time:.2f} seconds")
