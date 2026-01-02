import time
import logging
from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

def evaluate_model(pipeline, X_test, y_test, model_name, feature_engineering_name, results_dir, tune=False, cross_validation=False, X_train=None, y_train=None):
    """
    Evaluate the trained model and save results.
    
    Args:
        pipeline: Trained sklearn pipeline
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        feature_engineering_name: Name of feature engineering technique
        results_dir: Directory to save results
        tune: Whether hyperparameter tuning was applied
        cross_validation: Whether to perform cross-validation
        X_train: Training features (for cross-validation)
        y_train: Training labels (for cross-validation)
    
    Returns:
        tuple: (accuracy, cv_mean, elapsed_time)
    """
    logging.info(f"Evaluating model: {model_name} with feature engineering: {feature_engineering_name}")
    
    # Make predictions
    start_time = time.time()
    predictions = pipeline.predict(X_test)
    elapsed_time = time.time() - start_time
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, digits=4)

    # Perform cross-validation if requested
    if cross_validation and X_train is not None and y_train is not None:
        logging.info("Performing cross-validation on training data...")
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        logging.info(f"Cross-validation scores: {cv_scores}")
        logging.info(f"Cross-validation mean: {cv_mean:.4f} (+/- {cv_std:.4f})")
    else:
        # Use test set for cross-validation if training data not available
        cv_scores = cross_val_score(pipeline, X_test, y_test, cv=5, scoring='accuracy', n_jobs=-1)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        logging.info(f"Cross-validation on test set - mean: {cv_mean:.4f} (+/- {cv_std:.4f})")

    # Save the trained model
    model_filename = f"{model_name.replace(' ', '_').lower()}_{feature_engineering_name.replace(' ', '_').lower()}_model.pkl"
    model_path = results_dir / model_filename
    joblib.dump(pipeline, model_path)
    logging.info(f"Model saved as {model_filename}")

    # Save evaluation report
    report_file_path = results_dir / f"{model_name.replace(' ', '_').lower()}_{feature_engineering_name.replace(' ', '_').lower()}{'_tuned' if tune else ''}_report.txt"
    with open(report_file_path, 'w') as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Feature Engineering: {feature_engineering_name}\n")
        file.write(f"Hyperparameter Tuning: {'Applied' if tune else 'Not Applied'}\n")
        file.write(f"Cross-Validation: {'Applied' if cross_validation else 'Not Applied'}\n")
        file.write(f"Test Accuracy: {accuracy:.4f}\n")
        file.write(f"Cross-Validation Mean Accuracy: {cv_mean:.4f}\n")
        if cross_validation:
            file.write(f"Cross-Validation Std: {cv_std:.4f}\n")
        file.write(f"Prediction Time: {elapsed_time:.4f} seconds\n")
        file.write("\nClassification Report:\n")
        file.write(report)
    logging.info(f"Results saved to {report_file_path}")

    return accuracy, cv_mean, elapsed_time
