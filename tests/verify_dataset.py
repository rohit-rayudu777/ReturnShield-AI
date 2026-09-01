import pandas as pd
import numpy as np
from src.data.generator import generate_synthetic_data

def run_checks():
    csv_path = "C:\\Users\\rohit\\Documents\\ReturnShield_ai\\data\\returns.csv"
    print(f"Loading generated dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 2. Print shape
    print("\n--- 2. Dataset Shape ---")
    print(f"Shape: {df.shape}")
    
    # 3. Print target class distribution
    print("\n--- 3. Target Class Distribution ---")
    dist = df["is_abusive_return"].value_counts(normalize=True)
    counts = df["is_abusive_return"].value_counts()
    for val, count in counts.items():
        print(f"Class {val}: {count} ({dist[val]*100:.2f}%)")
        
    # 4. Check missing values
    print("\n--- 4. Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")
    
    # 5. Check duplicate order IDs
    print("\n--- 5. Duplicate Order IDs ---")
    dupes = df["order_id"].duplicated().sum()
    print(f"Number of duplicate order IDs: {dupes}")
    
    # 6. Check numeric ranges
    print("\n--- 6. Numeric Ranges ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        print(f"{col:<25} | Min: {df[col].min():<10.2f} | Max: {df[col].max():<10.2f} | Mean: {df[col].mean():<10.2f}")
        
    # 7. Check categorical distributions
    print("\n--- 7. Categorical Distributions ---")
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if col in ["order_id", "customer_id", "timestamp"]:
            print(f"{col:<20} | Unique Count: {df[col].nunique()}")
            continue
        print(f"\nDistribution of {col}:")
        print(df[col].value_counts(normalize=True).round(4) * 100)
        
    # 8. Check timestamp range
    print("\n--- 8. Timestamp Range ---")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Min Timestamp: {df['timestamp'].min()}")
    print(f"Max Timestamp: {df['timestamp'].max()}")
    print(f"Span: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    
    # 9. Verify reproducibility using the same random seed
    print("\n--- 9. Verifying Reproducibility (Seed 42) ---")
    df_first = generate_synthetic_data(seed=42, num_returns_target=1000)
    df_second = generate_synthetic_data(seed=42, num_returns_target=1000)
    
    reproducible = df_first.equals(df_second)
    print(f"Do independent runs with seed 42 match? {reproducible}")
    
    # 10. Confirm that no target-derived feature is being used
    # Look at correlation with target to verify no perfect leakages
    print("\n--- 10. Correlation with target (is_abusive_return) ---")
    correlations = df.select_dtypes(include=[np.number]).corr()["is_abusive_return"].sort_values(ascending=False)
    print(correlations)

if __name__ == "__main__":
    run_checks()
