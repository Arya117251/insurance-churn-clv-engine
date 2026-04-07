import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split

print("=" * 70)
print("CUSTOMER RISK MATRIX ANALYSIS")
print("Combining Churn Risk + CLV Predictions for Strategic Segmentation")
print("=" * 70)

# Load both models
print("\nLoading models...")
churn_model = joblib.load('outputs/models/best_churn_model.pkl')
clv_model = joblib.load('outputs/models/best_clv_model.pkl')
print("+ Churn model loaded")
print("+ CLV model loaded")

# Load the data
print("\nLoading and preprocessing data...")
df = pd.read_csv('data/features/model_ready_features.csv')
print(f"Original shape: {df.shape}")

# Keep individual_id for output
individual_ids = df['individual_id'].copy()

# Sample 20% of the data (same as training)
sample_indices = df.sample(frac=0.2, random_state=42).index
df = df.loc[sample_indices]
individual_ids = individual_ids.loc[sample_indices].reset_index(drop=True)
df.reset_index(drop=True, inplace=True)
print(f"Shape after 20% sampling: {df.shape}")

# Store actual values before preprocessing
actual_churn = df['Churn'].copy()
actual_clv = df['simple_clv'].copy()

# Preprocessing - encode categoricals
categorical_cols = ['state', 'marital_status', 'home_market_value', 'tenure_bucket',
                    'age_bucket', 'life_stage', 'income']

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"Shape after encoding: {df.shape}")

# Fill null values
df.fillna(0, inplace=True)

# Drop individual_id
if 'individual_id' in df.columns:
    df.drop(columns=['individual_id'], inplace=True)

# Prepare features for CHURN model (drop less features)
churn_features_to_drop = ['Churn', 'simple_clv', 'days_tenure', 'age_in_years',
                          'curr_ann_amt', 'state_churn_rate', 'tenure_years',
                          'premium_per_tenure_year']
X_churn = df.drop(columns=[col for col in churn_features_to_drop if col in df.columns])

# Prepare features for CLV model (drop more features for leakage prevention)
clv_features_to_drop = ['simple_clv', 'Churn', 'curr_ann_amt', 'tenure_years',
                        'premium_per_tenure_year', 'days_tenure']
X_clv = df.drop(columns=[col for col in clv_features_to_drop if col in df.columns])

print(f"Churn model features: {X_churn.shape[1]}")
print(f"CLV model features: {X_clv.shape[1]}")

# Create train/test split (80/20) to use test set
# For churn model
_, X_churn_test, _, y_churn_test, _, ids_churn_test, _, actual_churn_test = train_test_split(
    X_churn, actual_churn, individual_ids, actual_churn,
    test_size=0.2, random_state=42, stratify=actual_churn
)

# For CLV model (same split but different features)
_, X_clv_test, _, y_clv_test, _, ids_clv_test, _, actual_clv_test = train_test_split(
    X_clv, actual_clv, individual_ids, actual_clv,
    test_size=0.2, random_state=42, stratify=actual_churn  # stratify on churn to match
)

print(f"\nTest set size: {len(X_churn_test)} customers")

# Make predictions
print("\nGenerating predictions...")
churn_probabilities = churn_model.predict_proba(X_churn_test.values)[:, 1]
predicted_clv = clv_model.predict(X_clv_test.values)

print(f"+ Churn predictions generated")
print(f"+ CLV predictions generated")

# Define thresholds
CHURN_THRESHOLD = 0.70  # Optimal threshold from churn model
CLV_THRESHOLD = 800     # Relative high CLV among at-risk customers

print(f"\nSegmentation Thresholds:")
print(f"  Churn Risk: {CHURN_THRESHOLD} (optimal F1 threshold)")
print(f"  CLV: ${CLV_THRESHOLD:,} (relative high among at-risk)")

# Create segments
segments = []
for i in range(len(X_churn_test)):
    churn_prob = churn_probabilities[i]
    clv = predicted_clv[i]

    if churn_prob >= CHURN_THRESHOLD and clv >= CLV_THRESHOLD:
        segment = "HIGH_CHURN_HIGH_CLV"
        label = "!! CRITICAL - Retain Immediately"
    elif churn_prob >= CHURN_THRESHOLD and clv < CLV_THRESHOLD:
        segment = "HIGH_CHURN_LOW_CLV"
        label = "!  At Risk - Low Priority"
    elif churn_prob < CHURN_THRESHOLD and clv >= CLV_THRESHOLD:
        segment = "LOW_CHURN_HIGH_CLV"
        label = "*  Champions - Maintain"
    else:
        segment = "LOW_CHURN_LOW_CLV"
        label = "+  Stable - Monitor"

    segments.append((segment, label))

# Create results dataframe
results_df = pd.DataFrame({
    'individual_id': ids_churn_test.values,
    'churn_prob': churn_probabilities,
    'predicted_clv': predicted_clv,
    'actual_clv': actual_clv_test.values,
    'actual_churn': actual_churn_test.values,
    'segment': [s[0] for s in segments],
    'churn_label': [s[1] for s in segments]
})

# Calculate segment statistics
print("\n" + "=" * 70)
print("CUSTOMER SEGMENT ANALYSIS")
print("=" * 70)

segment_summary = results_df.groupby('segment').agg({
    'individual_id': 'count',
    'churn_prob': 'mean',
    'predicted_clv': 'mean',
    'actual_clv': 'mean',
    'actual_churn': 'mean'
}).round(4)

segment_summary.columns = ['Customer_Count', 'Avg_Churn_Prob', 'Avg_Predicted_CLV',
                           'Avg_Actual_CLV', 'Actual_Churn_Rate']

# Add labels
segment_labels = {
    'HIGH_CHURN_HIGH_CLV': '!! CRITICAL - Retain Immediately',
    'HIGH_CHURN_LOW_CLV': '!  At Risk - Low Priority',
    'LOW_CHURN_HIGH_CLV': '*  Champions - Maintain',
    'LOW_CHURN_LOW_CLV': '+  Stable - Monitor'
}

print("\nSegment Summary:")
print("-" * 70)
for segment in segment_summary.index:
    stats = segment_summary.loc[segment]
    print(f"\n{segment_labels[segment]}")
    print(f"  Count: {int(stats['Customer_Count']):,} customers ({stats['Customer_Count']/len(results_df)*100:.1f}%)")
    print(f"  Avg Churn Probability: {stats['Avg_Churn_Prob']:.2%}")
    print(f"  Avg Predicted CLV: ${stats['Avg_Predicted_CLV']:,.2f}")
    print(f"  Avg Actual CLV: ${stats['Avg_Actual_CLV']:,.2f}")
    print(f"  Actual Churn Rate: {stats['Actual_Churn_Rate']:.2%}")

# Find top 5 critical customers (high churn + high CLV)
print("\n" + "=" * 70)
print("TOP 5 CRITICAL CUSTOMERS (Highest CLV among High Churn Risk)")
print("=" * 70)

critical_customers = results_df[results_df['segment'] == 'HIGH_CHURN_HIGH_CLV'].sort_values(
    'predicted_clv', ascending=False
).head(5)

print(f"\n{'Rank':<6} {'Customer ID':<15} {'Churn Prob':<12} {'Predicted CLV':<15} {'Actual CLV':<15} {'Status':<10}")
print("-" * 80)

for i, (idx, row) in enumerate(critical_customers.iterrows(), 1):
    status = "CHURNED" if row['actual_churn'] == 1 else "RETAINED"
    print(f"{i:<6} {int(row['individual_id']):<15} {row['churn_prob']:.2%}{'':<6} "
          f"${row['predicted_clv']:>10,.2f}    ${row['actual_clv']:>10,.2f}    {status:<10}")

# Calculate potential revenue at risk
if len(critical_customers) > 0:
    total_risk = critical_customers['predicted_clv'].sum()
    print(f"\n$$ Total CLV at Risk (Top 5): ${total_risk:,.2f}")
    print(f"## Average CLV of Critical Customers: ${critical_customers['predicted_clv'].mean():,.2f}")

# Save results
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

os.makedirs('outputs/analysis', exist_ok=True)

output_path = 'outputs/analysis/customer_segments.csv'
results_df.to_csv(output_path, index=False)
file_size = os.path.getsize(output_path)

print(f"+ Customer segments saved to: {output_path}")
print(f"  File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
print(f"  Total customers: {len(results_df):,}")
print(f"  Columns: {', '.join(results_df.columns)}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
