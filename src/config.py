"""
Configuration file for the malicious traffic detection ML pipeline.

This file contains:
- Directory paths for data and results
- Available machine learning models and their hyperparameter grids
- Available feature engineering techniques
"""
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'

# Machine learning models configuration
# Each model has a tuple: (model_instance, hyperparameter_grid)
# The hyperparameter grid is used for tuning when --tune-hyperparameters is enabled
# Note: Parameters must be prefixed with 'classifier__' because they are in a Pipeline
MODELS = {
    'Naive Bayes': (GaussianNB(), {
        'classifier__var_smoothing': np.logspace(-9, -7, 3)
    }),
    'Decision Tree': (DecisionTreeClassifier(random_state=42), {
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }),
    'Logistic Regression': (LogisticRegression(random_state=42, solver='saga', n_jobs=-1), {
        'classifier__C': np.logspace(-2, 2, 4),
        'classifier__tol': [1e-3, 1e-2]
    }),
    'Random Forest': (RandomForestClassifier(random_state=42), {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__max_features': ['sqrt', 'log2', None]
    })
}

# Feature engineering techniques configuration
# Maps technique names to their implementation identifiers
FEATURE_ENGINEERING = {
    'None': None,
    'Standard Scaling': 'standard_scaler',
    'Min-Max Scaling': 'minmax_scaler',
    'PCA': 'pca',
    'Polynomial Features': 'polynomial_features',
    'SelectKBest': 'select_k_best'
}
