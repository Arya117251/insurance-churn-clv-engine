import joblib
import pandas as pd

# Load the model
print("Loading model...")
model = joblib.load('outputs/models/best_churn_model.pkl')

# Check what feature names the model has
if hasattr(model, 'feature_names_in_'):
    print('\n=== Model feature_names_in_ (first 20) ===')
    for i, name in enumerate(model.feature_names_in_[:20]):
        print(f"{i}: {name}")
    print(f'\nTotal features in model: {len(model.feature_names_in_)}')
else:
    print('Model does not have feature_names_in_')

# Now let's see what the actual encoded features should look like
print('\n=== Loading actual features ===')
features_df = pd.read_csv('data/features/model_ready_features.csv', nrows=5)
print(f'Original columns: {features_df.columns.tolist()}')

# Drop non-feature columns
features_df = features_df.drop(columns=['individual_id', 'simple_clv', 'churn_lapse_ind'], errors='ignore')

# Apply one-hot encoding
categorical_cols = ['state', 'marital_status', 'home_market_value', 'tenure_bucket', 'age_bucket', 'life_stage', 'income']
categorical_cols = [col for col in categorical_cols if col in features_df.columns]

if len(categorical_cols) > 0:
    features_encoded = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)
    print(f'\n=== Encoded feature names (first 20) ===')
    for i, name in enumerate(features_encoded.columns.tolist()[:20]):
        print(f"{i}: {name}")
    print(f'\nTotal encoded features: {len(features_encoded.columns)}')
