#!/usr/bin/env python3
"""Auto-generated FastAPI prediction API."""
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib, pandas as pd, io, json, os

app = FastAPI(title="ML Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
pipeline = joblib.load("models/final_pipeline.pkl")
_fi_file = "models/feature_importance.json"
_feature_importance = json.load(open(_fi_file)) if os.path.exists(_fi_file) else []
_metrics  = json.load(open("models/metrics.json")) if os.path.exists("models/metrics.json") else {}
_rmse     = _metrics.get("rmse")
_ranges   = json.load(open("models/feature_ranges.json")) if os.path.exists("models/feature_ranges.json") else {}

class InputData(BaseModel):
    model_config = {"populate_by_name": True}
    id: Optional[float] = None
    age: Optional[float] = Field(None, alias='Age')
    annual_income: Optional[float] = Field(None, alias='Annual Income')
    number_of_dependents: Optional[float] = Field(None, alias='Number of Dependents')
    health_score: Optional[float] = Field(None, alias='Health Score')
    previous_claims: Optional[float] = Field(None, alias='Previous Claims')
    vehicle_age: Optional[float] = Field(None, alias='Vehicle Age')
    credit_score: Optional[float] = Field(None, alias='Credit Score')
    insurance_duration: Optional[float] = Field(None, alias='Insurance Duration')
    gender: Optional[str] = Field(None, alias='Gender')
    marital_status: Optional[str] = Field(None, alias='Marital Status')
    education_level: Optional[str] = Field(None, alias='Education Level')
    occupation: Optional[str] = Field(None, alias='Occupation')
    location: Optional[str] = Field(None, alias='Location')
    policy_type: Optional[str] = Field(None, alias='Policy Type')
    policy_start_date: Optional[str] = Field(None, alias='Policy Start Date')
    customer_feedback: Optional[str] = Field(None, alias='Customer Feedback')
    smoking_status: Optional[str] = Field(None, alias='Smoking Status')
    exercise_frequency: Optional[str] = Field(None, alias='Exercise Frequency')
    property_type: Optional[str] = Field(None, alias='Property Type')

@app.get("/")
def index():
    """Serve the prediction UI if index.html exists."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "ML Prediction API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.model_dump(by_alias=True)])
    pred = pipeline.predict(df)[0]
    result = {"prediction": float(pred), "feature_importance": _feature_importance}
    if _rmse:
        result["ci_lower"] = round(float(pred) - _rmse, 2)
        result["ci_upper"] = round(float(pred) + _rmse, 2)
    result["metrics"] = _metrics
    return result

@app.post("/predict/batch")
def predict_batch(data: List[InputData]):
    df = pd.DataFrame([d.model_dump(by_alias=True) for d in data])
    preds = pipeline.predict(df)
    return {"predictions": preds.tolist()}

@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    """Upload a CSV — returns predictions for every row."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")
    contents = await file.read()
    df_up = pd.read_csv(io.BytesIO(contents))
    preds_up = pipeline.predict(df_up)
    return {"count": len(preds_up), "predictions": preds_up.tolist()}

@app.get("/metrics")
def metrics_endpoint():
    return _metrics

@app.get("/ranges")
def ranges_endpoint():
    return _ranges

@app.get("/importance")
def importance():
    return {"feature_importance": _feature_importance}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
