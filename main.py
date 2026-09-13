from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import os 
import sys

# 1. Initialize App
app = FastAPI(title="Insurance Premium Predictor")

# 2. Load Model 
# (Ensure 'best_insurance_model.pkl' is in the same directory)
model_path = "data/best_insurance_model.pkl"

try:
    model = joblib.load("data/best_insurance_model.pkl")
    print(f"Successfully loaded model from {model_path}", flush=True)
except Exception as e:
    print(f"CRITICAL: Failed to load model from {model_path}: {e}", flush=True)
    sys.exit(1)

# 3. Define Request Schema
class InsuranceData(BaseModel):
    age_group: str
    gender: str
    bmi_category: str
    smoker: int
    children: int
    income_lpa: float
    city: str
    occupation: str
    type_policy: str
    type_product: str
    tenure_years: float
    reimbursement: int

# 4. Endpoints
@app.get("/")
def health_check():
    return {"status": "Active", "message": "API is running."}

@app.post("/predict")
def predict_premium(data: InsuranceData):
    try:
        # Convert JSON payload directly into a DataFrame
        # model.predict expects a 2D array or DataFrame matching training features
        input_df = pd.DataFrame([data.dict()])

        prediction = model.predict(input_df)

        return {
            "status": "success",
            "predicted_premium": round(float(prediction[0]), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)