# Quick Start Guide - MLOps Energy Prediction Project

## Prerequisites

- Windows with PowerShell
- UV package manager installed
- AWS account (for EC2 and S3)
- Neon PostgreSQL account

## Step-by-Step Execution

### 1. Environment Setup (Already Done ✓)

```powershell
cd c:\Users\krish\Desktop\Kiran-MLops
# UV is already installed and dependencies are synced
```

### 2. Data Cleaning (Already Done ✓)

```powershell
$env:Path = "C:\Users\krish\.local\bin;$env:Path"
uv run python src/data_cleaning.py
```

**Output:**
- `data/cleaned/train.csv` (6,371 rows)
- `data/cleaned/validate.csv` (6,371 rows)
- `data/cleaned/test.csv` (5,462 rows)
- `data/drift/production_data.csv` (3,641 rows)

### 3. Configure Environment Variables

Create `.env` file from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and add your credentials:
- AWS Access Key ID and Secret
- Neon PostgreSQL connection details
- EC2 Public IP (after setup)

### 4. Set Up Remote MLflow (Optional - Can Use Local First)

**Option A: Local MLflow (Quick Start)**
```powershell
# Start local MLflow server
$env:Path = "C:\Users\krish\.local\bin;$env:Path"
uv run mlflow ui --port 5000
```

**Option B: Remote MLflow (Production)**
Follow `mlflow_setup/setup_ec2.md` for complete AWS setup.

### 5. Run H2O AutoML

```powershell
$env:Path = "C:\Users\krish\.local\bin;$env:Path"
uv run python src/h2o_automl.py
```

This identifies top 3 model types (takes ~10 minutes).

### 6. Train Models with MLflow

**Before training, ensure MLflow is running!**

```powershell
# Terminal 1: Start MLflow (if using local)
uv run mlflow ui --port 5000

# Terminal 2: Train models
$env:Path = "C:\Users\krish\.local\bin;$env:Path"

# Model 1: XGBoost
uv run python src/train_model1.py

# Model 2: LightGBM
uv run python src/train_model2.py

# Model 3: Random Forest
uv run python src/train_model3.py
```

**Access MLflow UI:** http://localhost:5000

### 7. Deploy FastAPI

```powershell
$env:Path = "C:\Users\krish\.local\bin;$env:Path"
uv run uvicorn src.api.main:app --reload
```

**Access API:**
- Swagger UI: http://localhost:8000/docs
- API Root: http://localhost:8000

### 8. Test API

```powershell
# In a new terminal
$env:Path = "C:\Users\krish\.local\bin;$env:Path"
uv run python tests/test_api.py

# Or see curl examples
uv run python tests/test_api.py --curl
```

### 9. Run Drift Analysis

```powershell
$env:Path = "C:\Users\krish\.local\bin;$env:Path"
uv run python src/drift_analysis.py
```

Opens `drift_report.html` in browser.

## Project Structure Summary

```
Kiran-MLops/
├── data/
│   ├── raw/energydata_complete.csv          # Original dataset
│   ├── cleaned/                              # Train/val/test splits
│   └── drift/production_data.csv             # For drift analysis
├── src/
│   ├── data_cleaning.py                      # ✓ Data preprocessing
│   ├── h2o_automl.py                         # ✓ AutoML model selection
│   ├── train_model1.py                       # ✓ XGBoost training
│   ├── train_model2.py                       # ✓ LightGBM training
│   ├── train_model3.py                       # ✓ Random Forest training
│   ├── drift_analysis.py                     # ✓ Drift detection
│   └── api/main.py                           # ✓ FastAPI application
├── mlflow_setup/
│   ├── mlflow_config.py                      # ✓ MLflow configuration
│   └── setup_ec2.md                          # ✓ EC2 setup guide
├── tests/test_api.py                         # ✓ API testing
├── pyproject.toml                            # ✓ UV dependencies
└── README.md                                 # ✓ Documentation
```

## Common Issues & Solutions

### Issue: UV not recognized
```powershell
$env:Path = "C:\Users\krish\.local\bin;$env:Path"
```

### Issue: MLflow connection failed
- Check if MLflow server is running
- Verify MLFLOW_TRACKING_URI in `.env`
- For local: use `http://localhost:5000`

### Issue: Models not loading in API
- Train models first using `train_model*.py` scripts
- Check MLflow Model Registry has registered models
- Verify model names match in API code

### Issue: H2O cluster initialization fails
- Close any existing H2O instances
- Restart PowerShell
- Try again

## Next Steps for Submission

1. **Take Screenshots:**
   - MLflow experiment tracking UI
   - MLflow Model Registry
   - FastAPI Swagger UI
   - Drift analysis report
   - Save to `screenshots/` folder

2. **Create Model Comparison Table:**
   - Extract metrics from MLflow UI
   - Update README.md with comparison
   - Document champion model selection

3. **Prepare GitHub Repository:**
   - Initialize git: `git init`
   - Add files: `git add .`
   - Commit: `git commit -m "MLOps final project"`
   - Push to GitHub

4. **Record Video Presentation (6 minutes):**
   - Project overview (1 min)
   - Data cleaning & splitting (1 min)
   - Model training & MLflow (2 min)
   - API deployment & testing (1 min)
   - Drift analysis (1 min)

## Rubric Checklist

- [x] Dataset Selection (5%) - UCI Energy Data
- [x] Data Cleaning (15%) - Handled missing values, outliers
- [x] Time-Based Splitting (10%) - 35%/35%/30% chronological
- [x] AutoML Analysis (10%) - H2O AutoML script
- [x] Manual Training (15%) - 3 models with MLflow
- [ ] Remote MLflow Setup (10%) - EC2 guide provided
- [ ] Model Registry (10%) - Scripts ready, needs execution
- [x] FastAPI Deployment (10%) - 3 endpoints implemented
- [x] Drift Analysis (5%) - Evidently integration
- [x] Code Quality (3%) - UV-based, well-structured
- [ ] Video Presentation (7%) - To be recorded

## Support

For issues or questions:
1. Check `README.md` for detailed documentation
2. Review `mlflow_setup/setup_ec2.md` for MLflow setup
3. Check MLflow UI for experiment logs
4. Review API docs at http://localhost:8000/docs
