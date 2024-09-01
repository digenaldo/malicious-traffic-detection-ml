from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import time
import joblib
import logging

def create_pipeline(model, feature_engineering=None):
    steps = []
    if feature_engineering:
        steps.append(('feature_engineering', feature_engineering))
    steps.append(('classifier', model))
    return Pipeline(steps=steps)

def tune_hyperparameters(pipeline, param_grid, X_train, y_train):
    logging.info("Starting hyperparameter tuning...")
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    logging.info(f"Best parameters found: {random_search.best_params_}")
    return random_search.best_estimator_, random_search.best_params_
