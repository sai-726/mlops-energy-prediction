"""
Model 3: Random Forest Training with MLflow

This script trains a Random Forest model and logs everything to MLflow.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from mlflow_setup.mlflow_config import setup_mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
TRAIN_PATH = Path("data/cleaned/train.csv")
VAL_PATH = Path("data/cleaned/validate.csv")
TEST_PATH = Path("data/cleaned/test.csv")

def load_data():
    """Load and prepare data"""
    print("\n[1/6] Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    exclude_cols = ['date', 'rv1', 'rv2']
    target = 'Appliances'
    
    X_train = train_df.drop(columns=exclude_cols + [target])
    y_train = train_df[target]
    X_val = val_df.drop(columns=exclude_cols + [target])
    y_val = val_df[target]
    X_test = test_df.drop(columns=exclude_cols + [target])
    y_test = test_df[target]
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(X_train, y_train):
    """Train Random Forest model"""
    print("\n[2/6] Training Random Forest model...")
    
    params = {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    print(f"[OK] Model trained with {params['n_estimators']} trees")
    return model, params

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate model on all splits"""
    print("\n[3/6] Evaluating model...")
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'val_r2': r2_score(y_val, y_val_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    print("\nMetrics:")
    print(f"  Train - RMSE: {metrics['train_rmse']:.4f}, MAE: {metrics['train_mae']:.4f}, R²: {metrics['train_r2']:.4f}")
    print(f"  Val   - RMSE: {metrics['val_rmse']:.4f}, MAE: {metrics['val_mae']:.4f}, R²: {metrics['val_r2']:.4f}")
    print(f"  Test  - RMSE: {metrics['test_rmse']:.4f}, MAE: {metrics['test_mae']:.4f}, R²: {metrics['test_r2']:.4f}")
    
    return metrics, y_test_pred

def create_plots(model, X_train, y_test, y_test_pred):
    """Create visualization plots"""
    print("\n[4/6] Creating plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    sns.barplot(data=importance_df, y='feature', x='importance', ax=axes[0])
    axes[0].set_title('Top 15 Feature Importances')
    axes[0].set_xlabel('Importance')
    
    # Predictions vs Actual
    axes[1].scatter(y_test, y_test_pred, alpha=0.5)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title('Predictions vs Actual (Test Set)')
    
    plt.tight_layout()
    plot_path = "random_forest_plots.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"[OK] Plots saved to {plot_path}")
    return plot_path

def main():
    print("=" * 70)
    print("MODEL 3: RANDOM FOREST TRAINING")
    print("=" * 70)
    
    mlflow = setup_mlflow()
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    with mlflow.start_run(run_name="random_forest_model"):
        model, params = train_model(X_train, y_train)
        metrics, y_test_pred = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)
        plot_path = create_plots(model, X_train, y_test, y_test_pred)
        
        print("\n[5/6] Logging to MLflow...")
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(plot_path)
        
        pred_df = pd.DataFrame({'actual': y_test, 'predicted': y_test_pred})
        pred_path = "random_forest_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(pred_path)
        
        print("[OK] Logged to MLflow")
        
        print("\n[6/6] Registering model...")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, "RandomForest_Energy_Model")
        print("[OK] Model registered")
    
    print("\n" + "=" * 70)
    print("[OK] RANDOM FOREST TRAINING COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()
