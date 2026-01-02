from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import time
import logging

def create_pipeline(model, feature_engineering=None):
    """
    Create a sklearn pipeline with optional feature engineering step.
    
    Args:
        model: The machine learning model to use
        feature_engineering: Optional feature engineering transformer
    
    Returns:
        sklearn Pipeline object
    """
    steps = []
    if feature_engineering:
        steps.append(('feature_engineering', feature_engineering))
    steps.append(('classifier', model))
    return Pipeline(steps=steps)

def tune_hyperparameters(pipeline, param_grid, X_train, y_train):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    
    Args:
        pipeline: sklearn Pipeline to tune
        param_grid: Dictionary of hyperparameters to search
        X_train: Training features
        y_train: Training labels
    
    Returns:
        tuple: (best_estimator, best_params)
    """
    logging.info("Starting hyperparameter tuning...")
    logging.info(f"Parameter grid size: {len(param_grid)} parameters")
    
    try:
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
            scoring='accuracy',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        random_search.fit(X_train, y_train)
        logging.info(f"Best parameters found: {random_search.best_params_}")
        logging.info(f"Best cross-validation score: {random_search.best_score_:.4f}")
        return random_search.best_estimator_, random_search.best_params_
    except Exception as e:
        logging.error(f"Error during hyperparameter tuning: {str(e)}")
        raise
