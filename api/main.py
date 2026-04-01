from fastapi import FastAPI
import mlflow.pyfunc
import joblib

from schemas import PredictRequest, PredictResponse

mlflow.set_tracking_uri("mlruns")
app = FastAPI(title="Fraud Detection API")

model = mlflow.pyfunc.load_model("models:/xgbmodel@champion")
scaler = joblib.load("api/scaler.pkl")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # 1. Parse input into DataFrame
    # 2. Scale numeric features with loaded scaler
    # 3. Run model.predict_proba
    # 4. Return fraud probability + flag
    pass
