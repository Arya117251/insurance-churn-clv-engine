import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor

# Load the data
print("Loading data...")
df = pd.read_csv('data/features/model_ready_features.csv')
print(f"Original shape: {df.shape}")

# Sample 20% of the data for memory efficiency
df = df.sample(frac=0.2, random_state=42)
df.reset_index(drop=True, inplace=True)
print(f"Shape after 20% sampling: {df.shape}")

# Print all column names
print("\n" + "=" * 60)
print("ALL COLUMNS IN DATASET")
print("=" * 60)
print(f"Total columns: {len(df.columns)}")
print(f"Columns: {df.columns.tolist()}")

# Check for potential leakage features
print("\n" + "=" * 60)
print("POTENTIAL LEAKAGE FEATURES CHECK")
print("=" * 60)

tenure_cols = [col for col in df.columns if 'tenure' in col.lower()]
premium_cols = [col for col in df.columns if 'premium' in col.lower()]
amt_cols = [col for col in df.columns if 'amt' in col.lower()]

print(f"\nColumns containing 'tenure': {tenure_cols}")
print(f"Columns containing 'premium': {premium_cols}")
print(f"Columns containing 'amt': {amt_cols}")

all_potential_leakage = set(tenure_cols + premium_cols + amt_cols)
print(f"\nAll potential leakage features: {sorted(all_potential_leakage)}")
print("=" * 60)

# Analyze simple_clv column
print("\n" + "=" * 60)
print("SIMPLE_CLV ANALYSIS")
print("=" * 60)

if 'simple_clv' in df.columns:
    print(f"Null values in simple_clv: {df['simple_clv'].isnull().sum()}")
    print(f"Min CLV: ${df['simple_clv'].min():,.2f}")
    print(f"Max CLV: ${df['simple_clv'].max():,.2f}")
    print(f"Mean CLV: ${df['simple_clv'].mean():,.2f}")
    print(f"Median CLV: ${df['simple_clv'].median():,.2f}")
    print(f"Std Dev: ${df['simple_clv'].std():,.2f}")
else:
    print("ERROR: simple_clv column not found in dataset!")
    print(f"Available columns: {df.columns.tolist()}")

print("=" * 60)

# Preprocessing
print("\n" + "=" * 60)
print("PREPROCESSING DATA")
print("=" * 60)

# Identify categorical columns to encode
categorical_cols = ['state', 'marital_status', 'home_market_value', 'tenure_bucket',
                    'age_bucket', 'life_stage', 'income']

print(f"Categorical columns to encode: {categorical_cols}")

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"Shape after encoding: {df.shape}")

# Fill null values with 0
null_count = df.isnull().sum().sum()
print(f"Total null values before fillna: {null_count}")
df.fillna(0, inplace=True)
print("Filled nulls with 0")

# Drop individual_id
if 'individual_id' in df.columns:
    df.drop(columns=['individual_id'], inplace=True)
    print("Dropped individual_id")

# Set target variable (simple_clv) and drop from features
y = df['simple_clv'].copy()

# Drop target and leakage columns from features
columns_to_drop = ['simple_clv']
if 'churn_lapse_ind' in df.columns:
    columns_to_drop.append('churn_lapse_ind')
if 'Churn' in df.columns:
    columns_to_drop.append('Churn')

# Drop CLV formula components to avoid leakage
leakage_features = ['curr_ann_amt', 'tenure_years', 'premium_per_tenure_year', 'days_tenure']
for feature in leakage_features:
    if feature in df.columns:
        columns_to_drop.append(feature)

X = df.drop(columns=columns_to_drop)
print(f"Dropped from features: {columns_to_drop}")
print(f"Final feature count: {X.shape[1]}")

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n" + "=" * 60)
print("TRAIN/TEST SPLIT SUMMARY")
print("=" * 60)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train mean: ${y_train.mean():,.2f}")
print(f"y_test mean: ${y_test.mean():,.2f}")
print(f"y_train std: ${y_train.std():,.2f}")
print(f"y_test std: ${y_test.std():,.2f}")
print("=" * 60)

# Train LightGBM Regressor
print("\n" + "=" * 60)
print("TRAINING LIGHTGBM REGRESSOR")
print("=" * 60)

# Create and train model
lgbm_model = LGBMRegressor(random_state=42, verbose=-1)
print("Training model...")
lgbm_model.fit(X_train, y_train)
print("Model trained successfully!")

# Make predictions on test set
y_pred = lgbm_model.predict(X_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
mae = np.mean(np.abs(y_test - y_pred))
print(f"RMSE (Root Mean Squared Error): ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: ${mae:,.2f}")
print("=" * 60)

# Save model and metrics
print("\n" + "=" * 60)
print("SAVING CLV MODEL AND METRICS")
print("=" * 60)

# Create output directory if it doesn't exist
os.makedirs('outputs/models', exist_ok=True)

# Save the LightGBM model
model_path = 'outputs/models/best_clv_model.pkl'
joblib.dump(lgbm_model, model_path)
model_size = os.path.getsize(model_path)
print(f"Model saved to: {model_path}")
print(f"Model file size: {model_size:,} bytes ({model_size / 1024:.2f} KB)")

# Save model metrics
metrics = {
    'rmse': float(rmse),
    'r2_score': float(r2),
    'mae': float(mae),
    'features_count': X_train.shape[1],
    'train_samples': X_train.shape[0],
    'test_samples': X_test.shape[0]
}

metrics_path = 'outputs/models/clv_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
metrics_size = os.path.getsize(metrics_path)
print(f"Metrics saved to: {metrics_path}")
print(f"Metrics file size: {metrics_size:,} bytes")
print("=" * 60)
