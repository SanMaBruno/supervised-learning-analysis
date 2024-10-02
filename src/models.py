import joblib
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_lda(X_train, y_train):
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model

def train_all_models(X_train, y_train):
    models = {
        'Naive Bayes': train_naive_bayes(X_train, y_train),
        'LDA': train_lda(X_train, y_train),
        'Logistic Regression': train_logistic_regression(X_train, y_train)
    }
    return models

def save_model(model, model_name, project_path):
    models_path = os.path.join(project_path, 'results', 'models')
    os.makedirs(models_path, exist_ok=True)
    file_path = os.path.join(models_path, f'{model_name.lower().replace(" ", "_")}_model.pkl')
    joblib.dump(model, file_path)
    print(f'Modelo {model_name} guardado en: {file_path}')
