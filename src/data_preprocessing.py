# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_breast_cancer_data(file_path):
    # Definir los nombres de las columnas
    column_names = [
        'ID', 'Diagnosis', 'Radius_mean', 'Texture_mean', 'Perimeter_mean', 'Area_mean', 
        'Smoothness_mean', 'Compactness_mean', 'Concavity_mean', 'Concave_points_mean', 
        'Symmetry_mean', 'Fractal_dimension_mean', 'Radius_se', 'Texture_se', 'Perimeter_se', 
        'Area_se', 'Smoothness_se', 'Compactness_se', 'Concavity_se', 'Concave_points_se', 
        'Symmetry_se', 'Fractal_dimension_se', 'Radius_worst', 'Texture_worst', 
        'Perimeter_worst', 'Area_worst', 'Smoothness_worst', 'Compactness_worst', 
        'Concavity_worst', 'Concave_points_worst', 'Symmetry_worst', 'Fractal_dimension_worst'
    ]

    # Cargar el dataset
    data = pd.read_csv(file_path, header=None, names=column_names)
    
    # Separar características (X) y etiqueta (y)
    X = data.drop(['ID', 'Diagnosis'], axis=1)
    y = data['Diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # 1 para maligno, 0 para benigno

    return X, y

def preprocess_data(X, y):
    # Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Estandarizar las características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
