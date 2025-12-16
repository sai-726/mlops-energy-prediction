# MLOps Final Project - Energy Consumption Prediction

A complete end-to-end MLOps pipeline for predicting appliance energy consumption using the UCI Energy Data dataset.

## Project Overview

This project demonstrates a production-ready MLOps workflow including:
- ✅ Data cleaning and time-based train/validation/test splitting
- ✅ H2O AutoML for algorithm selection
- ✅ Manual training of 3 models with MLflow experiment tracking
- ✅ Remote MLflow server (AWS EC2 + PostgreSQL + S3)
- ✅ FastAPI deployment with multiple model endpoints
- ✅ Drift detection using Evidently

## Dataset

**UCI Energy Data** - Appliance energy consumption prediction
- **Rows**: 19,737 time-series records (10-minute intervals)
- **Period**: January - May 2016
- **Target**: `Appliances` (energy consumption in Wh)
- **Features**: 27 features including temperature sensors (T1-T9), humidity sensors (RH_1-RH_9), and weather data

## Project Structure

```
Kiran-MLops/
├── data/
│   ├── raw/                    # Original dataset
│   ├── cleaned/                # Train/validation/test splits
│   └── drift/                  # Production simulation data
├── src/
│   ├── data_cleaning.py        # Data preprocessing
│   ├── h2o_automl.py          # H2O AutoML training
│   ├── train_model1.py        # Model 1 training
│   ├── train_model2.py        # Model 2 training
│   ├── train_model3.py        # Model 3 training
│   ├── drift_analysis.py      # Drift detection
│   └── api/                    # FastAPI application
├── mlflow_setup/              # MLflow configuration
├── screenshots/               # MLflow UI screenshots
└── tests/                     # API tests
```

## Installation

This project uses **UV** for Python package management (not conda/venv).

### 1. Install UV

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

```bash
cd Kiran-MLops
uv sync
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

Edit `.env` with your AWS and Neon PostgreSQL credentials.

## Usage

### Step 1: Data Cleaning and Splitting

```bash
uv run python src/data_cleaning.py
```

This creates:
- `data/cleaned/train.csv` (35% - first chronological portion)
- `data/cleaned/validate.csv` (35% - middle portion)
- `data/cleaned/test.csv` (30% - final portion)
- `data/drift/production_data.csv` (last 20% for drift analysis)

### Step 2: H2O AutoML

```bash
uv run python src/h2o_automl.py
```

Identifies top 3 model types based on RMSE.

### Step 3: Manual Model Training

```bash
uv run python src/train_model1.py
uv run python src/train_model2.py
uv run python src/train_model3.py
```

Each script logs to MLflow with metrics (RMSE, MAE, R²) and artifacts.

### Step 4: FastAPI Deployment

```bash
uv run uvicorn src.api.main:app --reload
```

Access API at: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Endpoints: `/predict_model1`, `/predict_model2`, `/predict_model3`

### Step 5: Drift Analysis

```bash
uv run python src/drift_analysis.py
```

Generates drift detection report using Evidently.

## MLflow Setup

See `mlflow_setup/setup_ec2.md` for detailed instructions on:
1. Setting up EC2 instance
2. Configuring Neon PostgreSQL
3. Creating S3 bucket
4. Starting MLflow tracking server

## Model Comparison

| Model | Train RMSE | Val RMSE | Test RMSE | Test MAE | Test R² | Status |
|-------|------------|----------|-----------|----------|---------|--------|
| **XGBoost** | 12.99 | 79.77 | 116.35 | 107.44 | -9.58 | ✅ Registered |
| **Gradient Boosting** | 16.87 | 76.40 | 96.96 | 85.09 | -6.35 | ✅ Registered |
| **Random Forest** | 16.37 | 47.49 | **49.89** | **44.24** | **-0.94** | 🏆 **Champion** |

**Champion Model:** Random Forest (promoted to Production)

**Selection Criteria:**
- ✅ Lowest test RMSE: 49.89 Wh
- ✅ Best drift stability: 4.65% performance degradation
- ✅ Most generalizable: Least overfitting on test data

**Note on Negative R²:** The negative R² values on validation/test sets are expected for time-series data with seasonal drift. Models trained on winter data (Jan-Feb) show degradation on spring data (Apr-May). RMSE is the primary metric for model selection in this regression task.

## API Examples

### Using curl

```bash
curl -X POST "http://localhost:8000/predict_model1" \
  -H "Content-Type: application/json" \
  -d '{
    "T1": 19.89, "RH_1": 47.6, "T2": 19.2, "RH_2": 44.8,
    "T3": 19.79, "RH_3": 44.73, "T4": 19.0, "RH_4": 45.57,
    "T5": 17.17, "RH_5": 55.2, "T6": 7.03, "RH_6": 84.26,
    "T7": 17.2, "RH_7": 41.63, "T8": 18.2, "RH_8": 48.9,
    "T9": 17.03, "RH_9": 45.53, "T_out": 6.6, "Press_mm_hg": 733.5,
    "RH_out": 92.0, "Windspeed": 7.0, "Visibility": 63.0, "Tdewpoint": 5.3
  }'
```

### Using Python

```python
import requests

data = {
    "T1": 19.89, "RH_1": 47.6, "T2": 19.2, "RH_2": 44.8,
    # ... other features
}

response = requests.post("http://localhost:8000/predict_model1", json=data)
print(response.json())
```

## Screenshots

See `screenshots/` directory for:
- MLflow experiment tracking
- Model registry
- Drift analysis reports

## Team

- Sai Kiran Kanduri
- Mohana Thota

## License

Educational project for MLOps course.
