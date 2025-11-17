# ==========================================
# 🏦 Bank Customer Churn Prediction - predict.py
# ==========================================

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load Model and Scaler

model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

#  Define FastAPI app

app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="Predict whether a customer will churn based on demographic and account data.",
    version="1.0"
)

#  Define Input Schema (for validation)

class CustomerInput(BaseModel):
    credit_score: float
    gender: int               # 0 = Female, 1 = Male
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: int          # 1 = Has credit card, 0 = No
    active_member: int        # 1 = Active, 0 = Inactive
    estimated_salary: float
    country_germany: int      # 1 = Yes, 0 = No
    country_spain: int        # 1 = Yes, 0 = No

# Define prediction endpoint
@app.post("/predict")
def predict_customer_churn(data: CustomerInput):
    # Convert input to array
    input_data = np.array([[data.credit_score, 
                            data.gender, 
                            data.age, 
                            data.tenure, 
                            data.balance,
                            data.products_number, 
                            data.credit_card, 
                            data.active_member,
                            data.estimated_salary, 
                            data.country_germany, 
                            data.country_spain]])

    # Scale features
    input_scaled = scaler.transform(input_data)

    # Predict churn
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 3)
    }

# Root endpoint

@app.get("/")
def read_root():
    return {"message": "Welcome to Bank Customer Churn Prediction API. Visit /docs for testing."}
