"""
Data Cleaning and Time-Based Splitting Script

This script:
1. Loads the raw energy consumption data
2. Handles missing values and outliers
3. Converts date column to datetime
4. Performs time-based train/validation/test split (35%/35%/30%)
5. Saves cleaned splits to data/cleaned/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
RAW_DATA_PATH = Path("data/raw/energydata_complete.csv")
CLEANED_DIR = Path("data/cleaned")
DRIFT_DIR = Path("data/drift")

# Create directories if they don't exist
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
DRIFT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load raw data from CSV"""
    print(f"Loading data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    return df


def check_missing_values(df):
    """Check for missing values"""
    print("\n=== Checking Missing Values ===")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("[OK] No missing values found")
    else:
        print("Missing values found:")
        print(missing[missing > 0])
    return df


def handle_outliers(df):
    """Remove outliers using IQR method for numerical columns"""
    print("\n=== Handling Outliers ===")
    
    # Select numerical columns (exclude date)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Store original count
    original_count = len(df)
    
    # Apply IQR method to target variable only (Appliances)
    # We'll be conservative and only remove extreme outliers
    target_col = 'Appliances'
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Use 3*IQR for more conservative outlier removal
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    print(f"Target variable ({target_col}) bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Filter outliers
    df_clean = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
    
    removed_count = original_count - len(df_clean)
    print(f"[OK] Removed {removed_count} outlier rows ({removed_count/original_count*100:.2f}%)")
    print(f"Remaining rows: {len(df_clean)}")
    
    return df_clean


def convert_date_column(df):
    """Convert date column to datetime and sort"""
    print("\n=== Converting Date Column ===")
    
    # Convert to datetime
    df['date'] = pd.to_datetime(df['date'])
    print(f"[OK] Converted 'date' column to datetime")
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    print(f"[OK] Sorted data by date")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def time_based_split(df):
    """
    Split data chronologically:
    - First 35% → Training
    - Next 35% → Validation
    - Final 30% → Test
    """
    print("\n=== Time-Based Splitting ===")
    
    total_rows = len(df)
    
    # Calculate split indices
    train_end = int(total_rows * 0.35)
    val_end = int(total_rows * 0.70)  # 35% + 35% = 70%
    
    # Split data
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"Total rows: {total_rows}")
    print(f"Training set: {len(train_df)} rows ({len(train_df)/total_rows*100:.1f}%)")
    print(f"  Date range: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Validation set: {len(val_df)} rows ({len(val_df)/total_rows*100:.1f}%)")
    print(f"  Date range: {val_df['date'].min()} to {val_df['date'].max()}")
    print(f"Test set: {len(test_df)} rows ({len(test_df)/total_rows*100:.1f}%)")
    print(f"  Date range: {test_df['date'].min()} to {test_df['date'].max()}")
    
    return train_df, val_df, test_df


def create_drift_data(df):
    """
    Create production simulation data from the last 20% of the dataset
    This will be used for drift analysis
    """
    print("\n=== Creating Drift/Production Data ===")
    
    total_rows = len(df)
    drift_start = int(total_rows * 0.80)  # Last 20%
    
    drift_df = df.iloc[drift_start:].copy()
    print(f"Production simulation data: {len(drift_df)} rows ({len(drift_df)/total_rows*100:.1f}%)")
    print(f"  Date range: {drift_df['date'].min()} to {drift_df['date'].max()}")
    
    return drift_df


def save_splits(train_df, val_df, test_df, drift_df):
    """Save cleaned splits to CSV files"""
    print("\n=== Saving Cleaned Data ===")
    
    train_path = CLEANED_DIR / "train.csv"
    val_path = CLEANED_DIR / "validate.csv"
    test_path = CLEANED_DIR / "test.csv"
    drift_path = DRIFT_DIR / "production_data.csv"
    
    train_df.to_csv(train_path, index=False)
    print(f"[OK] Saved training data to {train_path}")
    
    val_df.to_csv(val_path, index=False)
    print(f"[OK] Saved validation data to {val_path}")
    
    test_df.to_csv(test_path, index=False)
    print(f"[OK] Saved test data to {test_path}")
    
    drift_df.to_csv(drift_path, index=False)
    print(f"[OK] Saved production simulation data to {drift_path}")


def print_summary_statistics(train_df, val_df, test_df):
    """Print summary statistics for each split"""
    print("\n=== Summary Statistics ===")
    
    print("\nTarget Variable (Appliances) Statistics:")
    print(f"Training   - Mean: {train_df['Appliances'].mean():.2f}, Std: {train_df['Appliances'].std():.2f}")
    print(f"Validation - Mean: {val_df['Appliances'].mean():.2f}, Std: {val_df['Appliances'].std():.2f}")
    print(f"Test       - Mean: {test_df['Appliances'].mean():.2f}, Std: {test_df['Appliances'].std():.2f}")


def main():
    """Main execution function"""
    print("=" * 70)
    print("DATA CLEANING AND TIME-BASED SPLITTING")
    print("=" * 70)
    
    # Load data
    df = load_data()
    
    # Check missing values
    df = check_missing_values(df)
    
    # Convert date column
    df = convert_date_column(df)
    
    # Handle outliers
    df = handle_outliers(df)
    
    # Time-based split
    train_df, val_df, test_df = time_based_split(df)
    
    # Create drift data
    drift_df = create_drift_data(df)
    
    # Save splits
    save_splits(train_df, val_df, test_df, drift_df)
    
    # Print summary statistics
    print_summary_statistics(train_df, val_df, test_df)
    
    print("\n" + "=" * 70)
    print("[OK] DATA CLEANING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
