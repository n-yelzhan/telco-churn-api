from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

THRESHOLD = 0.2
model = joblib.load("churn_model.joblib")

app = FastAPI(title="Telco Churn API")


class Client(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
                "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
                "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 170.0
            }]
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(c: Client):
    X = pd.DataFrame([c.model_dump()])
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "churn_probability": round(proba, 3),
        "call_client": proba >= THRESHOLD,
        "threshold": THRESHOLD,
    }