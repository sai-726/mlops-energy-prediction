"""
Drift Analysis using NannyML

This script analyzes data and prediction drift using NannyML library.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from mlflow_setup.mlflow_config import setup_mlflow
import mlflow

# Paths
TEST_PATH = Path("data/cleaned/test.csv")
PROD_PATH = Path("data/drift/production_data.csv")

def load_data():
    """Load reference and production data"""
    print("\n[1/4] Loading data...")
    
    reference_df = pd.read_csv(TEST_PATH)
    production_df = pd.read_csv(PROD_PATH)
    
    print(f"Reference data: {len(reference_df)} rows")
    print(f"Production data: {len(production_df)} rows")
    
    return reference_df, production_df

def prepare_data_for_nannyml(reference_df, production_df):
    """Prepare data in NannyML format"""
    print("\n[2/4] Preparing data for NannyML...")
    
    # Exclude columns
    exclude_cols = ['date', 'rv1', 'rv2']
    target = 'Appliances'
    
    # Add identifier and timestamp columns required by NannyML
    reference_df = reference_df.copy()
    production_df = production_df.copy()
    
    reference_df['identifier'] = range(len(reference_df))
    production_df['identifier'] = range(len(reference_df), len(reference_df) + len(production_df))
    
    # Convert date to timestamp
    reference_df['timestamp'] = pd.to_datetime(reference_df['date'])
    production_df['timestamp'] = pd.to_datetime(production_df['date'])
    
    # Mark reference vs production
    reference_df['partition'] = 'reference'
    production_df['partition'] = 'analysis'
    
    # Combine
    combined_df = pd.concat([reference_df, production_df], ignore_index=True)
    
    # Get feature columns
    feature_cols = [col for col in combined_df.columns 
                   if col not in exclude_cols + [target, 'identifier', 'timestamp', 'partition']]
    
    print(f"[OK] Prepared {len(feature_cols)} features for analysis")
    
    return combined_df, feature_cols, target

def analyze_with_nannyml(combined_df, feature_cols, target):
    """Analyze drift using NannyML"""
    print("\n[3/4] Analyzing drift with NannyML...")
    
    try:
        import nannyml as nml
        
        # Univariate drift detection
        calc = nml.UnivariateDriftCalculator(
            column_names=feature_cols,
            timestamp_column_name='timestamp',
            chunk_size=1000
        )
        
        # Fit on reference data
        reference_data = combined_df[combined_df['partition'] == 'reference']
        calc.fit(reference_data)
        
        # Calculate drift on production data
        analysis_data = combined_df[combined_df['partition'] == 'analysis']
        results = calc.calculate(analysis_data)
        
        # Get drift results
        drift_results = results.to_df()
        
        print(f"[OK] Drift analysis complete")
        print(f"[OK] Analyzed {len(feature_cols)} features")
        
        # Generate report
        figure = results.plot()
        figure.write_html("nannyml_drift_report.html")
        print(f"[OK] Report saved to nannyml_drift_report.html")
        
        return drift_results
        
    except ImportError as e:
        print(f"[ERROR] NannyML not available: {e}")
        print("[INFO] Falling back to statistical drift detection...")
        return perform_statistical_drift_analysis(combined_df, feature_cols, target)
    except Exception as e:
        print(f"[ERROR] NannyML analysis failed: {e}")
        print("[INFO] Falling back to statistical drift detection...")
        return perform_statistical_drift_analysis(combined_df, feature_cols, target)

def perform_statistical_drift_analysis(combined_df, feature_cols, target):
    """Fallback: Statistical drift analysis"""
    
    reference_data = combined_df[combined_df['partition'] == 'reference']
    analysis_data = combined_df[combined_df['partition'] == 'analysis']
    
    drift_results = []
    
    for feature in feature_cols:
        ref_mean = reference_data[feature].mean()
        ref_std = reference_data[feature].std()
        prod_mean = analysis_data[feature].mean()
        prod_std = analysis_data[feature].std()
        
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
    
    print(f"[OK] Statistical analysis complete")
    print(f"[OK] Detected drift in {drifted_features} features ({drifted_features/len(feature_cols)*100:.1f}%)")
    
    # Save results
    drift_df.to_csv("drift_analysis_results.csv", index=False)
    print(f"[OK] Results saved to drift_analysis_results.csv")
    
    return drift_df

def analyze_model_performance(combined_df, target):
    """Analyze model performance drift"""
    print("\n[4/4] Analyzing model performance drift...")
    
    mlflow_client = setup_mlflow()
    
    reference_data = combined_df[combined_df['partition'] == 'reference']
    analysis_data = combined_df[combined_df['partition'] == 'analysis']
    
    exclude_cols = ['date', 'rv1', 'rv2', 'identifier', 'timestamp', 'partition']
    
    X_ref = reference_data.drop(columns=exclude_cols + [target], errors='ignore')
    y_ref = reference_data[target]
    X_prod = analysis_data.drop(columns=exclude_cols + [target], errors='ignore')
    y_prod = analysis_data[target]
    
    models = {
        'XGBoost': 'XGBoost_Energy_Model',
        'GradientBoosting': 'GradientBoosting_Energy_Model',
        'RandomForest': 'RandomForest_Energy_Model'
    }
    
    results = []
    
    for model_name, registry_name in models.items():
        try:
            model = mlflow.pyfunc.load_model(f"models:/{registry_name}/latest")
            
            # Predictions
            y_ref_pred = model.predict(X_ref)
            y_prod_pred = model.predict(X_prod)
            
            # RMSE
            from sklearn.metrics import mean_squared_error
            ref_rmse = np.sqrt(mean_squared_error(y_ref, y_ref_pred))
            prod_rmse = np.sqrt(mean_squared_error(y_prod, y_prod_pred))
            
            rmse_change = abs((prod_rmse - ref_rmse) / ref_rmse * 100) if ref_rmse != 0 else 0
            
            results.append({
                'model': model_name,
                'ref_rmse': ref_rmse,
                'prod_rmse': prod_rmse,
                'rmse_change_%': rmse_change
            })
            
            print(f"[OK] {model_name}: {rmse_change:.2f}% RMSE change")
            
        except Exception as e:
            print(f"[WARNING] Could not analyze {model_name}: {e}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_performance_drift.csv", index=False)
    print(f"[OK] Model performance results saved")
    
    return results_df

def main():
    print("=" * 70)
    print("DRIFT ANALYSIS WITH NANNYML")
    print("=" * 70)
    
    # Load data
    reference_df, production_df = load_data()
    
    # Prepare data
    combined_df, feature_cols, target = prepare_data_for_nannyml(reference_df, production_df)
    
    # Analyze drift
    drift_results = analyze_with_nannyml(combined_df, feature_cols, target)
    
    # Analyze model performance
    perf_results = analyze_model_performance(combined_df, target)
    
    print("\n" + "=" * 70)
    print("[OK] DRIFT ANALYSIS COMPLETED")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - nannyml_drift_report.html (or drift_analysis_results.csv)")
    print("  - model_performance_drift.csv")
    print("=" * 70)

if __name__ == "__main__":
    main()
