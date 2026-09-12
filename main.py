from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn

# 1. Initialize App
app = FastAPI(title="Insurance Premium Predictor")

# 2. Load Model 
# (Ensure 'best_insurance_model.pkl' is in the same directory)
try:
    model = joblib.load("best_insurance_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")

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
