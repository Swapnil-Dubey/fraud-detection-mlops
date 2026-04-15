import os
import pickle
import tempfile

import mlflow
from mlflow import MlflowClient
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
import os
from evaluate import evaluate_model


mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("Experiments")

client = MlflowClient()

MODEL_NAME = "xgb-model"
ALIAS = "champion"


def log_pickle(name, obj, tmp):
    path = os.path.join(tmp, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    mlflow.log_artifact(path)


def train(X_train, y_train, X_test, y_test, X_val, y_val,
          majorly_null_cols, low_cardinality_cols, high_cardinality_cols, ohe, te):
    with mlflow.start_run():
        dummymodel = DummyClassifier(strategy='most_frequent')
        dummymodel.fit(X_train, y_train)
        mlflow.log_metrics(evaluate_model(dummymodel, X_test, y_test, "dummy"))
        mlflow.sklearn.log_model(dummymodel, "dummy")

    with mlflow.start_run():
        params_xgb = {
            "scale_pos_weight": 27.6,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "max_depth": 6,
            "n_estimators": 500,
            "learning_rate": 0.1,
            "early_stopping_rounds": 10,
            "seed": 42,
        }
        xgboostmodel = XGBClassifier(**params_xgb)
        xgboostmodel.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        evalmetrics = evaluate_model(xgboostmodel, X_test, y_test, "XGB")
        mlflow.log_params(params_xgb)
        mlflow.log_metrics(evalmetrics)

        with tempfile.TemporaryDirectory() as tmp:
            log_pickle("nullcols.pkl", majorly_null_cols, tmp)
            log_pickle("low_cardinality_cols.pkl", low_cardinality_cols, tmp)
            log_pickle("high_cardinality_cols.pkl", high_cardinality_cols, tmp)
            log_pickle("ohe.pkl", ohe, tmp)
            log_pickle("te.pkl", te, tmp)

        res = mlflow.xgboost.log_model(xgboostmodel, "xgb", registered_model_name=MODEL_NAME)

        try:
            champion = client.get_model_version_by_alias(name=MODEL_NAME, alias=ALIAS)
            champion_f2 = client.get_run(champion.run_id).data.metrics.get('f2')
            if champion_f2 < evalmetrics["f2"]:
                client.set_registered_model_alias(name=MODEL_NAME, alias=ALIAS, version=res.registered_model_version)
        except Exception:
            client.set_registered_model_alias(name=MODEL_NAME, alias=ALIAS, version=res.registered_model_version)


if __name__ == "__main__":
    def load(name):
        with open(f'../data/processed/{name}.pkl', 'rb') as f:
            return pickle.load(f)

    train(
        X_train=load('X_train'), 
        y_train=load('y_train'),
        X_test=load('X_test'),   
        y_test=load('y_test'),
        X_val=load('X_val'),     
        y_val=load('y_val'),
        majorly_null_cols=load('majorly_null_cols'),
        low_cardinality_cols=load('low_cardinality_cols'),
        high_cardinality_cols=load('high_cardinality_cols'),
        ohe=load('ohe'),
        te=load('te'),
    )
