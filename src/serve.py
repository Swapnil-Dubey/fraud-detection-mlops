import pickle
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from preprocess import apply_ohe, apply_target_encoder, build_features, drop_cols


class FraudPredictionRequest(BaseModel):
    model_config = {"extra": "allow"}
    TransactionAmt: float
    TransactionDT: int
    TransactionID: int


mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = mlflow.tracking.MlflowClient()
model = ohe = te = majorly_null_cols = best_threshold = low_cardinality_cols = high_cardinality_cols = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, ohe, te, majorly_null_cols, best_threshold, low_cardinality_cols, high_cardinality_cols

    model_name = "xgb-model"
    alias = "champion"
    model_version = client.get_model_version_by_alias(name=model_name, alias=alias)
    model = mlflow.xgboost.load_model(f"models:/{model_name}/{model_version.version}")

    def load_artifact(name):
        path = mlflow.artifacts.download_artifacts(run_id=model_version.run_id, artifact_path=name)
        with open(path, "rb") as f:
            return pickle.load(f)

    ohe = load_artifact("ohe.pkl")
    te = load_artifact("te.pkl")
    majorly_null_cols = load_artifact("nullcols.pkl")
    low_cardinality_cols = load_artifact("low_cardinality_cols.pkl")
    high_cardinality_cols = load_artifact("high_cardinality_cols.pkl")
    best_threshold = client.get_run(model_version.run_id).data.metrics.get("best_threshold")

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
def predict(req: FraudPredictionRequest):
    try:
        df = pd.DataFrame([req.model_dump()])
        df = df.apply(lambda col: pd.to_numeric(col, errors='ignore') if col.dtype == object else col) # type cast any numeric cols that got mistype casted as object due to nulls in the json
        df = build_features(df, majorly_null_cols)
        cols_to_drop = ["TransactionDT", "TransactionID", "DeviceInfo", "isFraud"] + majorly_null_cols
        df = drop_cols(df, cols_to_drop)
        df = apply_ohe(df, ohe, low_cardinality_cols)
        df = apply_target_encoder(df, te, high_cardinality_cols)

        df = df.reindex(columns=model.get_booster().feature_names)
        prob = float(model.predict_proba(df)[:, 1][0])
        label = "fraud" if prob > best_threshold else "not fraud"
        return {"prediction_probability": prob, "label": label}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
