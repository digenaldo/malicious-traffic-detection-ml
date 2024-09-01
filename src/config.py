from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'

MODELS = {
    'Naive Bayes': (GaussianNB(), {
        'var_smoothing': np.logspace(-9, -7, 3)
    }),
    'Decision Tree': (DecisionTreeClassifier(random_state=42), {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }),
    'Logistic Regression': (LogisticRegression(random_state=42, solver='saga', n_jobs=-1), {
        'C': np.logspace(-2, 2, 4),
        'tol': [1e-3, 1e-2]
    }),
    'Random Forest': (RandomForestClassifier(random_state=42), {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['auto', 'sqrt']
    })
}

FEATURE_ENGINEERING = {
    'None': None,
    'Standard Scaling': 'standard_scaler',
    'Min-Max Scaling': 'minmax_scaler',
    'PCA': 'pca',
    'Polynomial Features': 'polynomial_features',
    'SelectKBest': 'select_k_best'
}
