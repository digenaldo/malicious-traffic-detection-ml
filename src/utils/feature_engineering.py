from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def get_feature_engineering(technique_name):
    if technique_name == 'Standard Scaling':
        return StandardScaler()
    elif technique_name == 'Min-Max Scaling':
        return MinMaxScaler()
    elif technique_name == 'PCA':
        return PCA(n_components=5, random_state=42)
    elif technique_name == 'Polynomial Features':
        return PolynomialFeatures(degree=2, include_bias=False)
    elif technique_name == 'SelectKBest':
        return SelectKBest(score_func=f_classif, k=5)
    return None
