from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "customer_churn_model.joblib"
)

model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]
threshold = float(model_bundle["threshold"])
feature_names = model_bundle["feature_names"]


class CustomerData(BaseModel):
    credit_score: int = Field(ge=300, le=850)
    country: Literal["France", "Germany", "Spain"]
    gender: Literal["Female", "Male"]
    age: int = Field(ge=18, le=100)
    tenure: int = Field(ge=0, le=10)
    balance: float = Field(ge=0)
    products_number: Literal[1, 2, 3, 4]
    credit_card: Literal[0, 1]
    active_member: Literal[0, 1]
    estimated_salary: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "credit_score": 620,
                    "country": "Germany",
                    "gender": "Female",
                    "age": 50,
                    "tenure": 4,
                    "balance": 120000.0,
                    "products_number": 3,
                    "credit_card": 1,
                    "active_member": 0,
                    "estimated_salary": 90000.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float
    threshold: float
    predicted_churn: bool
    recommended_action: Literal[
        "contact",
        "no_contact",
    ]


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "API for estimating customer churn probability "
        "and prioritizing retention contacts."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "feature_count": len(feature_names),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(customer: CustomerData):
    customer_frame = pd.DataFrame(
        [customer.model_dump()]
    )

    customer_frame = customer_frame[feature_names]

    churn_probability = float(
        model.predict_proba(customer_frame)[0, 1]
    )

    predicted_churn = (
        churn_probability >= threshold
    )

    recommended_action = (
        "contact"
        if predicted_churn
        else "no_contact"
    )

    return PredictionResponse(
        churn_probability=round(
            churn_probability,
            4,
        ),
        threshold=round(threshold, 4),
        predicted_churn=predicted_churn,
        recommended_action=recommended_action,
    )
