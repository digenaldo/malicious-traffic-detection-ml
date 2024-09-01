import time
import logging
from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

def evaluate_model(pipeline, X_test, y_test, model_name, feature_engineering_name, results_dir, tune=False):
    logging.info(f"Evaluating model: {model_name} with feature engineering: {feature_engineering_name}")
    start_time = time.time()
    predictions = pipeline.predict(X_test)
    elapsed_time = time.time() - start_time
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, digits=4)

    # Cross-validation scores
    cv_scores = cross_val_score(pipeline, X_test, y_test, cv=5, scoring='accuracy')
    cv_mean = np.mean(cv_scores)

    # Save the trained model
    model_filename = f"{model_name.replace(' ', '_').lower()}_{feature_engineering_name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(pipeline, results_dir / model_filename)
    logging.info(f"Model saved as {model_filename}")

    # Save individual report
    report_file_path = results_dir / f"{model_name.replace(' ', '_').lower()}_{feature_engineering_name.replace(' ', '_').lower()}{'_tuned' if tune else ''}_report.txt"
    with open(report_file_path, 'w') as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Feature Engineering: {feature_engineering_name}\n")
        file.write(f"Hyperparameter Tuning: {'Applied' if tune else 'Not Applied'}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Cross-Validation Mean Accuracy: {cv_mean:.4f}\n")
        file.write("\nClassification Report:\n")
        file.write(report)
    logging.info(f"Results saved to {report_file_path}")

    return accuracy, cv_mean, elapsed_time
