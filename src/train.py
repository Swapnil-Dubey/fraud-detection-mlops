import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from preprocess import load_data, build_features, scale_features
from evaluate import evaluate_model


TRANS_PATH = "data/raw/train_transaction.csv"
ID_PATH = "data/raw/train_identity.csv"
SCALER_PATH = "api/scaler.pkl"

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("fraud-detection")


def train():
    # 1. Load + preprocess

    # 2. Train/test split

    # 3. Scale numeric features

    # 4. Train baseline (LR) with MLflow run

    # 5. Train XGBoost with MLflow run

    # 6. Compare AUPRC, register champion
    pass


if __name__ == "__main__":
    train()
