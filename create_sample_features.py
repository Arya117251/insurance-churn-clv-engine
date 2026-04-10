"""
Create a sample of model_ready_features.csv for faster dashboard loading.
Samples 50,000 random rows to reduce file size for deployment.
"""
import pandas as pd
import os

# Paths
input_path = "data/features/model_ready_features.csv"
output_path = "data/features/model_ready_features_sample.csv"

# Check if input file exists
if not os.path.exists(input_path):
    print(f"ERROR: {input_path} not found!")
    exit(1)

print(f"Loading {input_path}...")
df = pd.read_csv(input_path)
print(f"Loaded {len(df):,} rows")

# Sample 50,000 random rows
sample_size = min(50000, len(df))  # In case file has fewer than 50k rows
print(f"Sampling {sample_size:,} random rows (random_state=42)...")
df_sample = df.sample(n=sample_size, random_state=42)

# Save sample
print(f"Saving to {output_path}...")
df_sample.to_csv(output_path, index=False)

# Verify
file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"Sample created successfully!")
print(f"   - Rows: {len(df_sample):,}")
print(f"   - Columns: {len(df_sample.columns)}")
print(f"   - File size: {file_size_mb:.2f} MB")
