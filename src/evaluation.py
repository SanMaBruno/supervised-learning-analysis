import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator
from typing import Dict, Tuple

def evaluate_model(model: BaseEstimator, X_test, y_test) -> Tuple[float, str]:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def evaluate_all_models(models: Dict[str, BaseEstimator], X_test, y_test) -> Dict[str, Dict[str, float]]:
    evaluation_results = {}
    for name, model in models.items():
        accuracy, report = evaluate_model(model, X_test, y_test)
        evaluation_results[name] = {
            'accuracy': accuracy,
            'classification_report': report
        }
    return evaluation_results

def plot_confusion_matrix(model: BaseEstimator, X_test, y_test, model_name: str) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.show()