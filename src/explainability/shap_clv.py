import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the saved CLV model
print("Loading CLV model...")
model = joblib.load('outputs/models/best_clv_model.pkl')
print("Model loaded successfully!")

# Load the data
print("\nLoading and preprocessing data...")
df = pd.read_csv('data/features/model_ready_features.csv')
print(f"Original shape: {df.shape}")

# Sample 20% of the data for memory efficiency
df = df.sample(frac=0.2, random_state=42)
df.reset_index(drop=True, inplace=True)
print(f"Shape after 20% sampling: {df.shape}")

# Preprocessing - same as train_clv.py
categorical_cols = ['state', 'marital_status', 'home_market_value', 'tenure_bucket',
                    'age_bucket', 'life_stage', 'income']

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"Shape after encoding: {df.shape}")

# Fill null values with 0
df.fillna(0, inplace=True)

# Drop individual_id
if 'individual_id' in df.columns:
    df.drop(columns=['individual_id'], inplace=True)

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
print(f"Final feature count: {X.shape[1]}")

# Train/test split (80/20) - same as train_clv.py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Test set shape: {X_test.shape}")

# Create SHAP TreeExplainer
print("\nCreating SHAP TreeExplainer for CLV model...")
explainer = shap.TreeExplainer(model)

# Compute SHAP values for test set
print("Computing SHAP values for test set...")
shap_values = explainer.shap_values(X_test.values)
print("SHAP values computed successfully!")

# Calculate mean absolute SHAP values for each feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Create feature importance dataframe
feature_names = X.columns.tolist()
shap_importance = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

# Print top 10 features
print("\n" + "=" * 60)
print("TOP 10 MOST IMPORTANT FEATURES (by Mean Absolute SHAP)")
print("=" * 60)
for i, row in enumerate(shap_importance.head(10).itertuples(), 1):
    print(f"{i}. {row.feature}: {row.mean_abs_shap:.4f}")
print("=" * 60)

# Save SHAP summary plot
print("\n" + "=" * 60)
print("SAVING SHAP CLV SUMMARY PLOT")
print("=" * 60)

# Create output directory if it doesn't exist
os.makedirs('outputs/explainability', exist_ok=True)

# Create SHAP summary plot (beeswarm)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test.values, feature_names=feature_names, show=False)
plt.tight_layout()

# Save the plot
plot_path = 'outputs/explainability/shap_clv_summary.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

plot_size = os.path.getsize(plot_path)
print(f"SHAP CLV summary plot saved to: {plot_path}")
print(f"Plot file size: {plot_size:,} bytes ({plot_size / 1024:.2f} KB)")
print("=" * 60)

# Function to explain individual customer CLV predictions
def explain_customer_clv(customer_idx, clv_label):
    """
    Explain CLV prediction for a specific customer from the test set.

    Args:
        customer_idx: Index of the customer in the test set
        clv_label: Label for this customer (e.g., "HIGH CLV", "MEDIUM CLV")
    """
    print("\n" + "=" * 60)
    print(f"CUSTOMER EXPLANATION - {clv_label} - Test Index: {customer_idx}")
    print("=" * 60)

    # Get customer's feature values
    customer_features = X_test.values[customer_idx]

    # Get customer's SHAP values
    customer_shap = shap_values[customer_idx]

    # Get predicted CLV
    customer_pred_clv = model.predict([customer_features])[0]

    # Get actual CLV
    actual_clv = y_test.iloc[customer_idx]

    # Create dataframe of feature contributions
    feature_contributions = pd.DataFrame({
        'feature': feature_names,
        'shap_value': customer_shap,
        'feature_value': customer_features
    })

    # Sort by absolute SHAP value
    feature_contributions['abs_shap'] = np.abs(feature_contributions['shap_value'])
    feature_contributions = feature_contributions.sort_values('abs_shap', ascending=False)

    # Print prediction summary
    print(f"\nPredicted CLV: ${customer_pred_clv:,.2f}")
    print(f"Actual CLV: ${actual_clv:,.2f}")
    print(f"Prediction Error: ${abs(customer_pred_clv - actual_clv):,.2f}")

    # Print top 5 SHAP features
    print(f"\nTop 5 Features Driving This CLV Prediction:")
    print(f"{'Feature':<40} {'SHAP Value':<15} {'Actual Value':<15}")
    print("-" * 70)

    for i, row in enumerate(feature_contributions.head(5).itertuples(), 1):
        direction = "+ INCREASE" if row.shap_value > 0 else "- DECREASE"
        print(f"{row.feature:<40} {row.shap_value:>8.2f} {direction:<11} {row.feature_value:>8.4f}")

    print("=" * 60)

# Find examples of high, medium, and low CLV customers
print("\n" + "=" * 60)
print("INDIVIDUAL CUSTOMER CLV EXPLANATIONS")
print("=" * 60)

# Get all predicted CLVs
all_predictions = model.predict(X_test.values)

# Find indices
high_clv_percentile = np.percentile(y_test, 90)
medium_clv_percentile = np.percentile(y_test, 50)
low_clv_percentile = np.percentile(y_test, 10)

# Find closest examples
high_clv_idx = np.argmin(np.abs(y_test - high_clv_percentile))
medium_clv_idx = np.argmin(np.abs(y_test - medium_clv_percentile))
low_clv_idx = np.argmin(np.abs(y_test - low_clv_percentile))

print(f"\nSelected Examples:")
print(f"- High CLV (90th percentile): ${y_test.iloc[high_clv_idx]:,.2f}")
print(f"- Medium CLV (50th percentile): ${y_test.iloc[medium_clv_idx]:,.2f}")
print(f"- Low CLV (10th percentile): ${y_test.iloc[low_clv_idx]:,.2f}")

# Explain each customer
explain_customer_clv(high_clv_idx, "HIGH CLV (90th percentile)")
explain_customer_clv(medium_clv_idx, "MEDIUM CLV (50th percentile)")
explain_customer_clv(low_clv_idx, "LOW CLV (10th percentile)")
