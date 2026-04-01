import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
mlflow.set_tracking_uri("../mlruns")


#Load and preprocess data

df = pd.read_csv('../data/raw/creditcard.csv')
X = df.drop("Class", axis = 1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify = y)

rscaler = RobustScaler()

X_train.loc[:, ["Amount", "Time"]] = rscaler.fit_transform(X_train[["Amount", "Time"]])
X_test.loc[:, ["Amount", "Time"]] = rscaler.transform(X_test[["Amount", "Time"]])
#Train Baseline (LR) model with MLflow

with mlflow.start_run():
    mlflow.log_param("model_name", "Logistic Regression Baseline")
    model = LogisticRegression(class_weight = "balanced")#balanced because for production system, when model trains on more data, the manual hardcoded split isn't the best idea
    model.
    mlflow.log_metric("AUPRC", ...)


#Train XGBoost with MLflow

#compare and register best model