import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load the model-ready features
df = pd.read_csv('data/features/model_ready_features.csv')
print(f"Shape of df: {df.shape}")

# Identify all categorical columns automatically
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Add 'income' to categorical columns (it's a binned range string)
if 'income' not in categorical_cols and 'income' in df.columns:
    categorical_cols.append('income')

print(f"Categorical columns found: {categorical_cols}")

# Remove target column if present
if 'churn_lapse_ind' in categorical_cols:
    categorical_cols.remove('churn_lapse_ind')

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"Shape after encoding: {df.shape}")

# Check and handle missing values
print(df.isnull().sum()[df.isnull().sum() > 0])
df.fillna(0, inplace=True)
print("Missing values filled with 0")

# Drop individual_id column
df.drop(columns=['individual_id'], inplace=True)
print("Dropped individual_id column")

# Drop redundant raw features
df.drop(columns=['days_tenure', 'age_in_years', 'curr_ann_amt'], inplace=True)
print("Dropped redundant raw features")

# Drop leaky/redundant features
df.drop(columns=['state_churn_rate', 'tenure_years', 'simple_clv', 'premium_per_tenure_year'], inplace=True)
print("Dropped leaky/redundant features")

# Sample 20% of the data for memory efficiency
df = df.sample(frac=0.2, random_state=42)
df.reset_index(drop=True, inplace=True)
print("Sampled to 20% for memory efficiency")
print(f"New shape: {df.shape}")

# Define target and features
y = df['Churn']
print(f"Unique values in y: {y.unique()}")
print(f"Value counts:\n{y.value_counts()}")
print(f"y dtype: {y.dtype}")
X = df.drop(columns=['Churn'])

# Print churn rate
print(f"Churn rate: {y.mean():.4f}")

# Print feature names
print(f"Feature names: {X.columns.tolist()}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled using StandardScaler")

# Create cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create and evaluate Logistic Regression model
lr_model = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
lr_scores = cross_val_score(lr_model, X_scaled, y, cv=cv, scoring='average_precision')
print(f"Logistic Regression PR-AUC: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")

# Create and evaluate XGBoost model
xgb_model = XGBClassifier(scale_pos_weight=7.7, random_state=42, eval_metric='logloss')
xgb_scores = cross_val_score(xgb_model, X.values, y, cv=cv, scoring='average_precision')
print(f"XGBoost PR-AUC: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")

# Create and evaluate LightGBM model
lgbm_model = LGBMClassifier(is_unbalance=True, random_state=42, verbose=-1)
lgbm_scores = cross_val_score(lgbm_model, X.values, y, cv=cv, scoring='average_precision')
print(f"LightGBM PR-AUC: {lgbm_scores.mean():.4f} ± {lgbm_scores.std():.4f}")

# Model comparison summary
print()
print("=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"Logistic Regression: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
print(f"XGBoost: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
print(f"LightGBM: {lgbm_scores.mean():.4f} ± {lgbm_scores.std():.4f}")

# Find best model
model_scores = {
    'Logistic Regression': lr_scores.mean(),
    'XGBoost': xgb_scores.mean(),
    'LightGBM': lgbm_scores.mean()
}
best_model = max(model_scores, key=model_scores.get)
best_score = model_scores[best_model]
print(f"Best Model: {best_model} with PR-AUC: {best_score:.4f}")

# Threshold tuning for LightGBM (best model)
print("\n" + "=" * 60)
print("THRESHOLD TUNING FOR LIGHTGBM")
print("=" * 60)

# Split data for threshold tuning
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=42, stratify=y)

# Train LightGBM on training set
lgbm_tuning = LGBMClassifier(is_unbalance=True, random_state=42, verbose=-1)
lgbm_tuning.fit(X_train, y_train)

# Get predicted probabilities on test set
y_pred_proba = lgbm_tuning.predict_proba(X_test)[:, 1]

# Sweep thresholds from 0.1 to 0.9 in steps of 0.05
thresholds = np.arange(0.1, 0.91, 0.05)
results = []

for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)

    # Calculate metrics for churn class (class=1)
    # True positives, false positives, false negatives
    tp = np.sum((y_test == 1) & (y_pred_threshold == 1))
    fp = np.sum((y_test == 0) & (y_pred_threshold == 1))
    fn = np.sum((y_test == 1) & (y_pred_threshold == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# Print table
print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 48)
for result in results:
    print(f"{result['threshold']:<12.2f} {result['precision']:<12.4f} {result['recall']:<12.4f} {result['f1']:<12.4f}")

# Find best threshold by F1
best_result = max(results, key=lambda x: x['f1'])
print("\n" + "=" * 60)
print(f"Best Threshold: {best_result['threshold']:.2f}")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  Recall: {best_result['recall']:.4f}")
print(f"  F1-Score: {best_result['f1']:.4f}")
print("=" * 60)

# Save the best model and optimal threshold
print("\n" + "=" * 60)
print("SAVING MODEL AND THRESHOLD")
print("=" * 60)

# Create output directory if it doesn't exist
os.makedirs('outputs/models', exist_ok=True)

# Save the LightGBM model
model_path = 'outputs/models/best_churn_model.pkl'
joblib.dump(lgbm_tuning, model_path)
model_size = os.path.getsize(model_path)
print(f"Model saved to: {model_path}")
print(f"Model file size: {model_size:,} bytes ({model_size / 1024:.2f} KB)")

# Save optimal threshold and metrics
threshold_info = {
    'optimal_threshold': best_result['threshold'],
    'precision': best_result['precision'],
    'recall': best_result['recall'],
    'f1_score': best_result['f1']
}

threshold_path = 'outputs/models/optimal_threshold.json'
with open(threshold_path, 'w') as f:
    json.dump(threshold_info, f, indent=4)
threshold_size = os.path.getsize(threshold_path)
print(f"Threshold info saved to: {threshold_path}")
print(f"Threshold file size: {threshold_size:,} bytes")
print("=" * 60)

# Fit models on full data to get feature importances
xgb_model.fit(X.values, y)
xgb_importances = xgb_model.feature_importances_

lgbm_model.fit(X.values, y)
lgbm_importances = lgbm_model.feature_importances_

# XGBoost feature importance
xgb_feature_pairs = list(zip(X.columns, xgb_importances))
xgb_feature_pairs.sort(key=lambda x: x[1], reverse=True)

print("\nXGBoost Top 10 Features:")
for i, (feature, importance) in enumerate(xgb_feature_pairs[:10], 1):
    print(f"{i}. {feature}: {importance:.4f}")

# LightGBM feature importance
lgbm_feature_pairs = list(zip(X.columns, lgbm_importances))
lgbm_feature_pairs.sort(key=lambda x: x[1], reverse=True)

print("\nLightGBM Top 10 Features:")
for i, (feature, importance) in enumerate(lgbm_feature_pairs[:10], 1):
    print(f"{i}. {feature}: {importance:.4f}")
