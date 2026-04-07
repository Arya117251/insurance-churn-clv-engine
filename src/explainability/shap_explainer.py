import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the saved model
print("Loading best churn model...")
model = joblib.load('outputs/models/best_churn_model.pkl')
print("Model loaded successfully!")

# Load the data
print("\nLoading and preprocessing data...")
df = pd.read_csv('data/features/model_ready_features.csv')
print(f"Original shape: {df.shape}")

# Identify all categorical columns automatically
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Add 'income' to categorical columns (it's a binned range string)
if 'income' not in categorical_cols and 'income' in df.columns:
    categorical_cols.append('income')

# Remove target column if present
if 'churn_lapse_ind' in categorical_cols:
    categorical_cols.remove('churn_lapse_ind')

print(f"Categorical columns: {categorical_cols}")

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"Shape after encoding: {df.shape}")

# Handle missing values
df.fillna(0, inplace=True)

# Drop individual_id column
df.drop(columns=['individual_id'], inplace=True)

# Drop redundant raw features
df.drop(columns=['days_tenure', 'age_in_years', 'curr_ann_amt'], inplace=True)

# Drop leaky/redundant features
df.drop(columns=['state_churn_rate', 'tenure_years', 'simple_clv', 'premium_per_tenure_year'], inplace=True)

# Sample 20% of the data for memory efficiency
df = df.sample(frac=0.2, random_state=42)
df.reset_index(drop=True, inplace=True)
print(f"Shape after sampling: {df.shape}")

# Define target and features
y = df['Churn']
X = df.drop(columns=['Churn'])

# Split data (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=42, stratify=y)
print(f"Test set shape: {X_test.shape}")

# Create SHAP TreeExplainer
print("\nCreating SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)

# Compute SHAP values for test set
print("Computing SHAP values for test set...")
shap_values = explainer.shap_values(X_test)

# If shap_values is a list (multi-class), get values for churn class (class 1)
if isinstance(shap_values, list):
    shap_values_churn = shap_values[1]
else:
    shap_values_churn = shap_values

print("SHAP values computed successfully!")

# Calculate mean absolute SHAP values for each feature
mean_abs_shap = np.abs(shap_values_churn).mean(axis=0)

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
print("SAVING SHAP SUMMARY PLOT")
print("=" * 60)

# Create output directory if it doesn't exist
os.makedirs('outputs/explainability', exist_ok=True)

# Create SHAP summary plot (beeswarm)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_churn, X_test, feature_names=feature_names, show=False)
plt.tight_layout()

# Save the plot
plot_path = 'outputs/explainability/shap_summary.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

plot_size = os.path.getsize(plot_path)
print(f"SHAP summary plot saved to: {plot_path}")
print(f"Plot file size: {plot_size:,} bytes ({plot_size / 1024:.2f} KB)")
print("=" * 60)

# Function to explain individual customer predictions
def explain_customer(customer_idx):
    """
    Explain churn prediction for a specific customer from the test set.

    Args:
        customer_idx: Index of the customer in the test set (0 to len(X_test)-1)
    """
    print("\n" + "=" * 60)
    print(f"CUSTOMER EXPLANATION - Test Set Index: {customer_idx}")
    print("=" * 60)

    # Get customer's feature values
    customer_features = X_test[customer_idx]

    # Get customer's SHAP values
    customer_shap = shap_values_churn[customer_idx]

    # Get predicted probability
    customer_proba = model.predict_proba([customer_features])[0][1]

    # Apply threshold
    optimal_threshold = 0.70
    is_churn = customer_proba >= optimal_threshold

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
    print(f"\nPredicted Churn Probability: {customer_proba:.4f} ({customer_proba*100:.2f}%)")
    print(f"Churn Flag (threshold=0.70): {'YES - CHURN' if is_churn else 'NO - RETAIN'}")
    print(f"Actual Label: {'CHURNED' if y_test.iloc[customer_idx] == 1 else 'RETAINED'}")

    # Print top 5 SHAP features
    print(f"\nTop 5 Features Driving This Prediction:")
    print(f"{'Feature':<40} {'SHAP Value':<15} {'Actual Value':<15}")
    print("-" * 70)

    for i, row in enumerate(feature_contributions.head(5).itertuples(), 1):
        direction = "+ CHURN" if row.shap_value > 0 else "- RETAIN"
        print(f"{row.feature:<40} {row.shap_value:>8.4f} {direction:<9} {row.feature_value:>8.4f}")

    print("=" * 60)

# Explain individual customers
print("\n" + "=" * 60)
print("INDIVIDUAL CUSTOMER EXPLANATIONS")
print("=" * 60)

explain_customer(0)
explain_customer(5)

# Find first customer flagged as churn
print("\n" + "=" * 60)
print("FINDING FIRST CUSTOMER FLAGGED AS CHURN (>= 0.70)")
print("=" * 60)

# Get predictions for all test customers
all_probas = model.predict_proba(X_test)[:, 1]

# Find first customer with probability >= 0.70
churn_threshold = 0.70
churn_customers = np.where(all_probas >= churn_threshold)[0]

if len(churn_customers) > 0:
    first_churn_idx = churn_customers[0]
    print(f"Found {len(churn_customers)} customers flagged as churn")
    print(f"First churn customer at test set index: {first_churn_idx}")
    explain_customer(first_churn_idx)
else:
    print("No customers flagged as churn at threshold 0.70")
