"""
Drift Analysis - Simple Version

This script analyzes data and prediction drift without external libraries.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from mlflow_setup.mlflow_config import setup_mlflow
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
TEST_PATH = Path("data/cleaned/test.csv")
PROD_PATH = Path("data/drift/production_data.csv")
REPORT_PATH = Path("drift_analysis_report.html")

def load_data():
    """Load reference and production data"""
    print("\n[1/5] Loading data...")
    
    reference_df = pd.read_csv(TEST_PATH)
    production_df = pd.read_csv(PROD_PATH)
    
    print(f"Reference data: {len(reference_df)} rows")
    print(f"Production data: {len(production_df)} rows")
    
    return reference_df, production_df

def load_models():
    """Load all trained models"""
    print("\n[2/5] Loading models from MLflow...")
    
    mlflow_client = setup_mlflow()
    
    models = {}
    try:
        models['xgboost'] = mlflow.pyfunc.load_model("models:/XGBoost_Energy_Model/latest")
        print("[OK] Loaded XGBoost model")
    except Exception as e:
        print(f"[WARNING] XGBoost model not found: {e}")
    
    try:
        models['gradient_boosting'] = mlflow.pyfunc.load_model("models:/GradientBoosting_Energy_Model/latest")
        print("[OK] Loaded Gradient Boosting model")
    except Exception as e:
        print(f"[WARNING] Gradient Boosting model not found: {e}")
    
    try:
        models['random_forest'] = mlflow.pyfunc.load_model("models:/RandomForest_Energy_Model/latest")
        print("[OK] Loaded Random Forest model")
    except Exception as e:
        print(f"[WARNING] Random Forest model not found: {e}")
    
    return models

def analyze_data_drift(reference_df, production_df):
    """Analyze data drift using statistical tests"""
    print("\n[3/5] Analyzing data drift...")
    
    exclude_cols = ['date', 'rv1', 'rv2', 'Appliances']
    features = [col for col in reference_df.columns if col not in exclude_cols]
    
    drift_results = []
    
    for feature in features:
        ref_mean = reference_df[feature].mean()
        ref_std = reference_df[feature].std()
        prod_mean = production_df[feature].mean()
        prod_std = production_df[feature].std()
        
        # Calculate percentage change
        mean_change = abs((prod_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0
        std_change = abs((prod_std - ref_std) / ref_std * 100) if ref_std != 0 else 0
        
        drift_detected = mean_change > 10 or std_change > 10  # 10% threshold
        
        drift_results.append({
            'feature': feature,
            'ref_mean': ref_mean,
            'prod_mean': prod_mean,
            'mean_change_%': mean_change,
            'drift_detected': drift_detected
        })
    
    drift_df = pd.DataFrame(drift_results)
    drifted_features = drift_df[drift_df['drift_detected']].shape[0]
    
    print(f"[OK] Analyzed {len(features)} features")
    print(f"[OK] Detected drift in {drifted_features} features ({drifted_features/len(features)*100:.1f}%)")
    
    return drift_df

def analyze_prediction_drift(models, reference_df, production_df):
    """Analyze prediction drift"""
    print("\n[4/5] Analyzing prediction drift...")
    
    exclude_cols = ['date', 'rv1', 'rv2']
    target = 'Appliances'
    
    X_ref = reference_df.drop(columns=exclude_cols + [target], errors='ignore')
    y_ref = reference_df[target]
    X_prod = production_df.drop(columns=exclude_cols + [target], errors='ignore')
    y_prod = production_df[target]
    
    results = []
    
    for model_name, model in models.items():
        try:
            # Predictions on reference data
            y_ref_pred = model.predict(X_ref)
            ref_rmse = np.sqrt(mean_squared_error(y_ref, y_ref_pred))
            ref_mae = mean_absolute_error(y_ref, y_ref_pred)
            ref_r2 = r2_score(y_ref, y_ref_pred)
            
            # Predictions on production data
            y_prod_pred = model.predict(X_prod)
            prod_rmse = np.sqrt(mean_squared_error(y_prod, y_prod_pred))
            prod_mae = mean_absolute_error(y_prod, y_prod_pred)
            prod_r2 = r2_score(y_prod, y_prod_pred)
            
            # Calculate drift
            rmse_change = abs((prod_rmse - ref_rmse) / ref_rmse * 100) if ref_rmse != 0 else 0
            
            results.append({
                'model': model_name,
                'ref_rmse': ref_rmse,
                'prod_rmse': prod_rmse,
                'rmse_change_%': rmse_change,
                'ref_mae': ref_mae,
                'prod_mae': prod_mae,
                'ref_r2': ref_r2,
                'prod_r2': prod_r2
            })
            
            print(f"[OK] {model_name}: RMSE change = {rmse_change:.2f}%")
            
        except Exception as e:
            print(f"[ERROR] Failed to analyze {model_name}: {e}")
    
    return pd.DataFrame(results)

def create_visualizations(drift_df, pred_drift_df):
    """Create drift visualization plots"""
    print("\n[5/5] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Top drifted features
    top_drift = drift_df.nlargest(10, 'mean_change_%')
    axes[0, 0].barh(top_drift['feature'], top_drift['mean_change_%'])
    axes[0, 0].set_xlabel('Mean Change (%)')
    axes[0, 0].set_title('Top 10 Features with Highest Drift')
    axes[0, 0].axvline(x=10, color='r', linestyle='--', label='10% threshold')
    axes[0, 0].legend()
    
    # Plot 2: Drift detection summary
    drift_counts = drift_df['drift_detected'].value_counts()
    axes[0, 1].pie(drift_counts, labels=['No Drift', 'Drift Detected'], autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Data Drift Detection Summary')
    
    # Plot 3: Model performance comparison
    if not pred_drift_df.empty:
        x = np.arange(len(pred_drift_df))
        width = 0.35
        axes[1, 0].bar(x - width/2, pred_drift_df['ref_rmse'], width, label='Reference')
        axes[1, 0].bar(x + width/2, pred_drift_df['prod_rmse'], width, label='Production')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('Model Performance: Reference vs Production')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(pred_drift_df['model'], rotation=45)
        axes[1, 0].legend()
    
    # Plot 4: RMSE change percentage
    if not pred_drift_df.empty:
        axes[1, 1].bar(pred_drift_df['model'], pred_drift_df['rmse_change_%'])
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('RMSE Change (%)')
        axes[1, 1].set_title('Model Performance Degradation')
        axes[1, 1].axhline(y=20, color='r', linestyle='--', label='20% threshold')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = "drift_analysis_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Plots saved to {plot_path}")
    return plot_path

def generate_html_report(drift_df, pred_drift_df, plot_path):
    """Generate HTML report"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Drift Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .alert {{ padding: 15px; margin: 20px 0; border-radius: 5px; }}
            .warning {{ background-color: #fff3cd; border-left: 5px solid #ffc107; }}
            .success {{ background-color: #d4edda; border-left: 5px solid #28a745; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Model Drift Analysis Report</h1>
        <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="alert warning">
            <strong>Summary:</strong> Detected drift in {drift_df[drift_df['drift_detected']].shape[0]} out of {len(drift_df)} features ({drift_df[drift_df['drift_detected']].shape[0]/len(drift_df)*100:.1f}%)
        </div>
        
        <h2>1. Data Drift Analysis</h2>
        <p>Comparison of feature distributions between reference (test) and production data.</p>
        {drift_df.to_html(index=False, classes='table')}
        
        <h2>2. Prediction Drift Analysis</h2>
        <p>Model performance comparison on reference vs production data.</p>
        {pred_drift_df.to_html(index=False, classes='table') if not pred_drift_df.empty else '<p>No model predictions available.</p>'}
        
        <h2>3. Visualizations</h2>
        <img src="{plot_path}" alt="Drift Analysis Plots">
        
        <h2>4. Recommendations</h2>
        <ul>
            <li>Features with >10% mean change should be investigated</li>
            <li>Models with >20% RMSE degradation should be retrained</li>
            <li>Consider implementing automated drift monitoring</li>
            <li>Schedule regular model retraining based on drift patterns</li>
        </ul>
        
    </body>
    </html>
    """
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[OK] HTML report saved to {REPORT_PATH}")

def main():
    print("=" * 70)
    print("DRIFT ANALYSIS")
    print("=" * 70)
    
    # Load data
    reference_df, production_df = load_data()
    
    # Load models
    models = load_models()
    
    # Analyze data drift
    drift_df = analyze_data_drift(reference_df, production_df)
    
    # Analyze prediction drift
    pred_drift_df = analyze_prediction_drift(models, reference_df, production_df)
    
    # Create visualizations
    plot_path = create_visualizations(drift_df, pred_drift_df)
    
    # Generate HTML report
    generate_html_report(drift_df, pred_drift_df, plot_path)
    
    print("\n" + "=" * 70)
    print(f"[OK] DRIFT ANALYSIS COMPLETED")
    print(f"[OK] Open {REPORT_PATH} in your browser to view the report")
    print("=" * 70)

if __name__ == "__main__":
    main()
