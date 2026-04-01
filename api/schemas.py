from pydantic import BaseModel


class PredictRequest(BaseModel):
    # Transaction features will go here once column list is finalized
    pass


class PredictResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    shap_explanation: dict | None = None
