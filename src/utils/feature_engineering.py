from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import logging

def get_feature_engineering(technique_name):
    """
    Get the feature engineering transformer based on technique name.
    
    Args:
        technique_name: Name of the feature engineering technique
    
    Returns:
        sklearn transformer or None
    """
    if technique_name == 'Standard Scaling':
        logging.info("Using StandardScaler for feature engineering")
        return StandardScaler()
    elif technique_name == 'Min-Max Scaling':
        logging.info("Using MinMaxScaler for feature engineering")
        return MinMaxScaler()
    elif technique_name == 'PCA':
        logging.info("Using PCA (5 components) for feature engineering")
        return PCA(n_components=5, random_state=42)
    elif technique_name == 'Polynomial Features':
        logging.info("Using PolynomialFeatures (degree=2) for feature engineering")
        return PolynomialFeatures(degree=2, include_bias=False)
    elif technique_name == 'SelectKBest':
        logging.info("Using SelectKBest (k=5) for feature engineering")
        return SelectKBest(score_func=f_classif, k=5)
    elif technique_name == 'None':
        logging.info("No feature engineering applied")
        return None
    else:
        logging.warning(f"Unknown feature engineering technique: {technique_name}. Using None.")
        return None
