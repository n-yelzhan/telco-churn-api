# Telco Churn API

Prediction of customer churn for a telecom operator, served as a REST API with FastAPI and packaged in Docker.

The model answers two questions for the retention team: **how likely is this customer to leave** and **should we contact them**.

## Results

| | Value |
|---|---|
| Dataset | Telco Customer Churn (IBM / Kaggle), 7,043 customers, churn rate ~27% |
| Model | Logistic Regression in a scikit-learn Pipeline (scaling + one-hot encoding) |
| ROC-AUC (test) | **0.842** |
| LightGBM (for comparison) | 0.839, same quality, so the simpler and interpretable model was chosen |
| Decision threshold | **0.2** instead of the default 0.5, chosen by retention economics: +28% expected profit under the stated assumptions |

Key churn drivers: month-to-month contract, short tenure (first 6 months), fiber optic internet, electronic check payment, no online security / tech support.

Full analysis (EDA, data quality, threshold tuning, feature importance and business recommendations): [Kaggle notebook](https://www.kaggle.com/yelzhan).

## Architecture

```
client JSON  →  FastAPI + Pydantic validation  →  scikit-learn Pipeline  →  probability + decision
                                                  (StandardScaler + OneHotEncoder + LogisticRegression)
```

- `GET /health` — service health check
- `POST /predict` — returns churn probability and whether to contact the customer

## Run with Docker

```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```

Open http://127.0.0.1:8000/docs for the interactive Swagger UI.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
uvicorn main:app --reload
```

## Example

Request `POST /predict`:

```json
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
  "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 170.0
}
```

Response:

```json
{
  "churn_probability": 0.629,
  "call_client": true,
  "threshold": 0.2
}
```

A loyal customer (two-year contract, 60 months tenure, DSL, automatic bank transfer) gets `churn_probability: 0.023` and `call_client: false`.

## Project structure

```
telco-churn-api/
├── main.py               # FastAPI app: schema, model loading, endpoints
├── churn_model.joblib    # trained scikit-learn Pipeline
├── requirements.txt      # pinned dependencies (scikit-learn 1.6.1, pandas 2.3.3, joblib 1.6.0)
├── Dockerfile
├── .dockerignore
└── .gitignore
```

## Next steps

- Monitoring: logging of requests and prediction distribution
- Tests for the API (pytest)
- Model retraining pipeline
