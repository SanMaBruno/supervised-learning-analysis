import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def add_custom_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el pipeline completo de ingeniería de características.

    Args:
        X (pd.DataFrame): DataFrame con las características estandarizadas.

    Returns:
        pd.DataFrame: DataFrame con nuevas características añadidas.
    """
    X = create_custom_features(X)
    X = add_polynomial_features(X)
    return X

def create_custom_features(X: pd.DataFrame) -> pd.DataFrame:
    # Asegurarse de que las columnas necesarias existan antes de realizar operaciones
    if 'Radius_mean' in X.columns and 'Texture_mean' in X.columns:
        X['radius_texture_ratio'] = X['Radius_mean'] / (X['Texture_mean'] + 1e-5)
    if 'Area_mean' in X.columns and 'Perimeter_mean' in X.columns:
        X['area_perimeter_ratio'] = X['Area_mean'] / (X['Perimeter_mean'] + 1e-5)
    return X

def add_polynomial_features(X: pd.DataFrame) -> pd.DataFrame:
    # Asegurarse de usar solo las columnas numéricas para las características polinomiales
    if all(col in X.columns for col in ['Radius_mean', 'Texture_mean', 'Area_mean']):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(X[['Radius_mean', 'Texture_mean', 'Area_mean']])
        poly_feature_names = poly.get_feature_names_out(['Radius_mean', 'Texture_mean', 'Area_mean'])
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=X.index)
        X = pd.concat([X, poly_df], axis=1)
    else:
        print("Las columnas necesarias para las características polinomiales no están presentes.")
    return X
