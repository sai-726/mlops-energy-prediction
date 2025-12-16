"""
MLflow Configuration

This module handles MLflow configuration and setup.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "energy-consumption-prediction")

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "mlops-energy-artifacts")

# PostgreSQL Configuration
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "mlflow_db")

def get_postgres_uri():
    """Get PostgreSQL connection URI"""
    return f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

def setup_mlflow():
    """Setup MLflow tracking"""
    import mlflow
    
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Set experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    print(f"[OK] MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"[OK] MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    
    return mlflow
