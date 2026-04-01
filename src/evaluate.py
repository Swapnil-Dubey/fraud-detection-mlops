import shap
import mlflow
from sklearn.metrics import average_precision_score


def evaluate_model(model, X_test, y_test, model_name: str) -> float:
    """Compute AUPRC and log to active MLflow run."""
    pass


def log_shap(model, X_train, X_test, model_name: str):
    """Compute SHAP values and log summary plot as MLflow artifact."""
    pass
