from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import os

app = FastAPI(title="Student Performance Prediction API")

# -----------------------------
# Global model holder
# -----------------------------
model = None

# -----------------------------
# Startup event (KEY FIX)
# -----------------------------
@app.on_event("startup")
def load_ml_model():
    global model

    MLFLOW_TRACKING_URI = os.getenv(
        "MLFLOW_TRACKING_URI",
        "sqlite:///C:/Users/hp/Desktop/mlflow-student-performance/src/mlflow.db"
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("Loading MLflow model...")
    model = mlflow.pyfunc.load_model(
        "models:/StudentPerformanceModel/Production"
    )
    print("Model loaded successfully")

# -----------------------------
# Request schema
# -----------------------------
class StudentInput(BaseModel):
    Age: int
    Gender: str
    Class: str
    Study_Hours_Per_Day: float
    Attendance_Percentage: float
    Parental_Education: str
    Internet_Access: str
    Extracurricular_Activities: str
    Previous_Year_Score: float

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: StudentInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    return {
        "predicted_final_percentage": float(prediction[0])
    }
