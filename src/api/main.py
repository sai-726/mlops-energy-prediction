"""
FastAPI Application for Energy Consumption Prediction

This application serves 3 models via REST API endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import mlflow
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from mlflow_setup.mlflow_config import setup_mlflow

# Initialize FastAPI
app = FastAPI(
    title="Energy Consumption Prediction API",
    description="API for predicting appliance energy consumption using 3 different models",
    version="1.0.0"
)

# Input schema
class EnergyFeatures(BaseModel):
    T1: float = Field(..., description="Temperature in kitchen area")
    RH_1: float = Field(..., description="Humidity in kitchen area")
    T2: float = Field(..., description="Temperature in living room area")
    RH_2: float = Field(..., description="Humidity in living room area")
    T3: float = Field(..., description="Temperature in laundry room area")
    RH_3: float = Field(..., description="Humidity in laundry room area")
    T4: float = Field(..., description="Temperature in office room")
    RH_4: float = Field(..., description="Humidity in office room")
    T5: float = Field(..., description="Temperature in bathroom")
    RH_5: float = Field(..., description="Humidity in bathroom")
    T6: float = Field(..., description="Temperature outside building (north side)")
    RH_6: float = Field(..., description="Humidity outside building (north side)")
    T7: float = Field(..., description="Temperature in ironing room")
    RH_7: float = Field(..., description="Humidity in ironing room")
    T8: float = Field(..., description="Temperature in teenager room 2")
    RH_8: float = Field(..., description="Humidity in teenager room 2")
    T9: float = Field(..., description="Temperature in parents room")
    RH_9: float = Field(..., description="Humidity in parents room")
    T_out: float = Field(..., description="Temperature outside (from weather station)")
    Press_mm_hg: float = Field(..., description="Pressure (mm Hg)")
    RH_out: float = Field(..., description="Humidity outside (from weather station)")
    Windspeed: float = Field(..., description="Wind speed (m/s)")
    Visibility: float = Field(..., description="Visibility (km)")
    Tdewpoint: float = Field(..., description="Dew point temperature")
    lights: float = Field(default=0, description="Energy use of light fixtures (Wh)")

    class Config:
        json_schema_extra = {
            "example": {
                "T1": 19.89, "RH_1": 47.6, "T2": 19.2, "RH_2": 44.8,
                "T3": 19.79, "RH_3": 44.73, "T4": 19.0, "RH_4": 45.57,
                "T5": 17.17, "RH_5": 55.2, "T6": 7.03, "RH_6": 84.26,
                "T7": 17.2, "RH_7": 41.63, "T8": 18.2, "RH_8": 48.9,
                "T9": 17.03, "RH_9": 45.53, "T_out": 6.6, "Press_mm_hg": 733.5,
                "RH_out": 92.0, "Windspeed": 7.0, "Visibility": 63.0, "Tdewpoint": 5.3,
                "lights": 30
            }
        }

# Output schema
class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    prediction: float
    timestamp: str
    input_features: Dict[str, float]

# Global variables for models
model1 = None
model2 = None
model3 = None

@app.on_event("startup")
async def load_models():
    """Load all models from MLflow on startup"""
    global model1, model2, model3
    
    try:
        mlflow_client = setup_mlflow()
        
        # Load Model 1 (XGBoost)
        try:
            model1 = mlflow.pyfunc.load_model("models:/XGBoost_Energy_Model/latest")
            print("[OK] Loaded XGBoost model")
        except Exception as e:
            print(f"[WARNING] Could not load XGBoost model: {e}")
        
        # Load Model 2 (Gradient Boosting)
        try:
            model2 = mlflow.pyfunc.load_model("models:/GradientBoosting_Energy_Model/latest")
            print("[OK] Loaded Gradient Boosting model")
        except Exception as e:
            print(f"[WARNING] Could not load Gradient Boosting model: {e}")
        
        # Load Model 3 (Random Forest)
        try:
            model3 = mlflow.pyfunc.load_model("models:/RandomForest_Energy_Model/latest")
            print("[OK] Loaded Random Forest model")
        except Exception as e:
            print(f"[WARNING] Could not load Random Forest model: {e}")
    
    except Exception as e:
        print(f"[ERROR] Failed to setup MLflow: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Energy Consumption Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict_model1": "XGBoost predictions",
            "/predict_model2": "Gradient Boosting predictions",
            "/predict_model3": "Random Forest predictions",
            "/health": "Health check",
            "/models": "List available models"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = []
    if model1 is not None:
        models_loaded.append("xgboost")
    if model2 is not None:
        models_loaded.append("gradient_boosting")
    if model3 is not None:
        models_loaded.append("random_forest")
    
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models")
async def list_models():
    """List available models"""
    models = []
    if model1 is not None:
        models.append("XGBoost")
    if model2 is not None:
        models.append("GradientBoosting")
    if model3 is not None:
        models.append("RandomForest")
    
    return {
        "models": models,
        "count": len(models)
    }

@app.post("/predict_model1", response_model=PredictionResponse)
async def predict_model1(features: EnergyFeatures):
    """
    Predict energy consumption using XGBoost model
    """
    if model1 is None:
        raise HTTPException(status_code=503, detail="XGBoost model not loaded")
    
    # Convert to DataFrame with correct column order (matching training data)
    feature_order = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 
                     'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
                     'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint']
    input_df = pd.DataFrame([features.dict()])[feature_order]
    
    # Make prediction
    prediction = float(model1.predict(input_df)[0])
    
    return PredictionResponse(
        model_name="XGBoost",
        model_version="latest",
        prediction=prediction,
        timestamp=datetime.now().isoformat(),
        input_features=features.dict()
    )

@app.post("/predict_model2", response_model=PredictionResponse)
async def predict_model2(features: EnergyFeatures):
    """
    Predict energy consumption using Gradient Boosting model
    """
    if model2 is None:
        raise HTTPException(status_code=503, detail="Gradient Boosting model not loaded")
    
    # Convert to DataFrame with correct column order (matching training data)
    feature_order = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 
                     'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
                     'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint']
    input_df = pd.DataFrame([features.dict()])[feature_order]
    
    # Make prediction
    prediction = float(model2.predict(input_df)[0])
    
    return PredictionResponse(
        model_name="GradientBoosting",
        model_version="latest",
        prediction=prediction,
        timestamp=datetime.now().isoformat(),
        input_features=features.dict()
    )

@app.post("/predict_model3", response_model=PredictionResponse)
async def predict_model3(features: EnergyFeatures):
    """
    Predict energy consumption using Random Forest model
    """
    if model3 is None:
        raise HTTPException(status_code=503, detail="Random Forest model not loaded")
    
    # Convert to DataFrame with correct column order (matching training data)
    feature_order = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 
                     'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
                     'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint']
    input_df = pd.DataFrame([features.dict()])[feature_order]
    
    # Make prediction
    prediction = float(model3.predict(input_df)[0])
    
    return PredictionResponse(
        model_name="RandomForest",
        model_version="latest",
        prediction=prediction,
        timestamp=datetime.now().isoformat(),
        input_features=features.dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
