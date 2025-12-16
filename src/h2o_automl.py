"""
H2O AutoML Script

This script uses H2O AutoML to identify the top 3 model types for energy consumption prediction.
"""

import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from pathlib import Path

# Paths
TRAIN_PATH = Path("data/cleaned/train.csv")
OUTPUT_PATH = Path("h2o_automl_results.txt")

def main():
    print("=" * 70)
    print("H2O AUTOML - MODEL SELECTION")
    print("=" * 70)
    
    # Initialize H2O
    print("\n[1/5] Initializing H2O cluster...")
    h2o.init()
    
    # Load training data
    print("\n[2/5] Loading training data...")
    train = h2o.import_file(str(TRAIN_PATH))
    print(f"Loaded {train.nrow} rows and {train.ncol} columns")
    
    # Define target and features
    target = 'Appliances'
    exclude_cols = ['date', 'rv1', 'rv2']  # Exclude date and random variables
    features = [col for col in train.columns if col not in exclude_cols + [target]]
    
    print(f"\nTarget: {target}")
    print(f"Features ({len(features)}): {features[:5]}... (showing first 5)")
    
    # Run AutoML
    print("\n[3/5] Running H2O AutoML (max 10 minutes)...")
    print("This may take a while...")
    
    aml = H2OAutoML(
        max_runtime_secs=600,  # 10 minutes
        max_models=20,
        seed=42,
        sort_metric='RMSE'
    )
    
    aml.train(x=features, y=target, training_frame=train)
    
    # Get leaderboard
    print("\n[4/5] Analyzing results...")
    lb = aml.leaderboard
    lb_df = lb.as_data_frame()
    
    print("\n" + "=" * 70)
    print("TOP 10 MODELS LEADERBOARD")
    print("=" * 70)
    print(lb_df.head(10).to_string())
    
    # Extract top 3 model types
    print("\n" + "=" * 70)
    print("TOP 3 MODEL TYPES FOR MANUAL TRAINING")
    print("=" * 70)
    
    model_types = []
    for i in range(min(10, len(lb_df))):
        model_id = lb_df.iloc[i]['model_id']
        if 'StackedEnsemble' in model_id:
            model_type = 'StackedEnsemble'
        elif 'GBM' in model_id:
            model_type = 'GBM (Gradient Boosting Machine)'
        elif 'XGBoost' in model_id:
            model_type = 'XGBoost'
        elif 'DeepLearning' in model_id:
            model_type = 'Deep Learning'
        elif 'GLM' in model_id:
            model_type = 'GLM (Generalized Linear Model)'
        elif 'DRF' in model_id:
            model_type = 'DRF (Distributed Random Forest)'
        else:
            model_type = model_id.split('_')[0]
        
        if model_type not in model_types and model_type != 'StackedEnsemble':
            model_types.append(model_type)
            rmse = lb_df.iloc[i]['rmse']
            mae = lb_df.iloc[i]['mae']
            print(f"\n{len(model_types)}. {model_type}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   MAE: {mae:.4f}")
            
            if len(model_types) == 3:
                break
    
    # Save results
    print("\n[5/5] Saving results...")
    with open(OUTPUT_PATH, 'w') as f:
        f.write("H2O AutoML Results\n")
        f.write("=" * 70 + "\n\n")
        f.write("Top 3 Model Types for Manual Training:\n\n")
        for i, mt in enumerate(model_types, 1):
            f.write(f"{i}. {mt}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("Full Leaderboard:\n\n")
        f.write(lb_df.head(10).to_string())
    
    print(f"[OK] Results saved to {OUTPUT_PATH}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDED MODELS FOR MANUAL TRAINING:")
    print("1. XGBoost (xgboost library)")
    print("2. LightGBM (lightgbm library)")
    print("3. Random Forest (scikit-learn)")
    print("=" * 70)
    
    # Shutdown H2O
    h2o.cluster().shutdown()
    
    print("\n[OK] H2O AutoML completed successfully!")

if __name__ == "__main__":
    main()
