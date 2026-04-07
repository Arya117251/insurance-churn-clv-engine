import pandas as pd
import numpy as np
from typing import Tuple

# =============================================================================
# DATA LEAKAGE WARNING — READ BEFORE MODIFYING THIS FILE
# =============================================================================
# The following features were identified as data leakage and must NEVER be
# used as model inputs:
#
#   - acct_suspd_date  : Recorded AFTER churn occurs, not before. A suspended
#                        account is a consequence of churn, not a predictor.
#   - is_suspended     : Derived directly from acct_suspd_date. Same issue.
#   - risk_score       : Composite feature that includes is_suspended. Tainted.
#
# Timeline of events:
#   Customer stops paying → Account suspended → Churn label set to 1
#
# In production, we score ACTIVE customers who have no suspension record yet.
# Including these features would make the model useless in the real world —
# it would only "predict" churn for customers who have already churned.
#
# These columns are retained in the engineered dataset for audit/reference
# but are explicitly excluded at the modeling stage in src/models/churn_model.py
# =============================================================================


def create_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create tenure-based features."""
    df = df.copy()
    df['tenure_years'] = df['days_tenure'] / 365

    df['tenure_bucket'] = pd.cut(
        df['tenure_years'],
        bins=[-np.inf, 1, 3, 5, 10, np.inf],
        labels=['<1yr', '1-3yr', '3-5yr', '5-10yr', '10yr+']
    )

    df['is_new_customer'] = (df['tenure_years'] < 1).astype(int)

    return df


def create_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create financial and wealth-based features."""
    df = df.copy()

    # Income is already numeric
    df['income_midpoint'] = df['income']

    # Parse home_market_value ranges to midpoints
    df['hmv_midpoint'] = df['home_market_value'].apply(parse_hmv_range)

    # Ratios with safe division
    df['premium_to_income_ratio'] = np.where(
        df['income_midpoint'] > 0,
        df['curr_ann_amt'] / df['income_midpoint'],
        0
    )

    df['premium_per_tenure_year'] = np.where(
        df['tenure_years'] > 0,
        df['curr_ann_amt'] / df['tenure_years'],
        0
    )

    df['wealth_index'] = df['income_midpoint'].fillna(0) + df['hmv_midpoint'].fillna(0)

    return df


def parse_hmv_range(value: str) -> float:
    """Parse home market value string ranges to midpoint."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if 'Plus' in value or 'plus' in value:
        return 1000000.0

    if '-' in value:
        parts = value.split('-')
        try:
            lower = float(parts[0].strip())
            upper = float(parts[1].strip())
            return (lower + upper) / 2
        except:
            return np.nan

    return np.nan


def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create behavior and risk-based features."""
    df = df.copy()

    df['is_suspended'] = df['acct_suspd_date'].notna().astype(int)

    df['risk_score'] = (
        df['is_suspended'] +
        (df['good_credit'] == 0).astype(int) +
        (df['premium_to_income_ratio'] > 0.1).astype(int)
    )

    df['risk_score_clean'] = (
        (df['good_credit'] == 0).astype(int) +
        (df['premium_to_income_ratio'] > 0.1).astype(int)
    )

    return df


def create_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create demographic and life stage features."""
    df = df.copy()

    df['age_bucket'] = pd.cut(
        df['age_in_years'],
        bins=[-np.inf, 30, 45, 60, np.inf],
        labels=['<30', '30-45', '45-60', '60+']
    )

    # Create life stage
    df['life_stage'] = df.apply(create_life_stage, axis=1)

    # Stability score
    df['stability_score'] = (
        df['home_owner'].fillna(0).astype(int) +
        (df['length_of_residence'] > 5).astype(int) +
        (df['marital_status'] == 'Married').astype(int)
    )

    return df


def create_life_stage(row: pd.Series) -> str:
    """Create life stage categorical from age, marital status, and children."""
    age_bucket = str(row['age_bucket'])
    marital = str(row['marital_status']).lower()
    has_kids = row['has_children'] == 1

    age_map = {'<30': 'young', '30-45': 'middle', '45-60': 'mature', '60+': 'senior'}
    age_label = age_map.get(age_bucket, 'unknown')

    marital_label = 'married' if 'married' in marital else 'single'
    kids_label = 'kids' if has_kids else 'no_kids'

    return f"{age_label}_{marital_label}_{kids_label}"


def create_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create location-based features including target encoding."""
    df = df.copy()

    # Calculate state-level churn rate
    state_churn = df.groupby('state')['Churn'].mean().to_dict()
    df['state_churn_rate'] = df['state'].map(state_churn)

    return df


def create_clv_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple CLV proxy features."""
    df = df.copy()

    df['simple_clv'] = df['curr_ann_amt'] * df['tenure_years']

    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply all feature engineering steps.

    Returns:
        Tuple of (original_df, engineered_df)
    """
    print("Starting feature engineering...")
    original_shape = df.shape

    # Apply all feature engineering
    df = create_tenure_features(df)
    df = create_financial_features(df)
    df = create_behavioral_features(df)
    df = create_demographic_features(df)
    df = create_location_features(df)
    df = create_clv_features(df)

    # Drop specified columns
    columns_to_drop = [
        'latitude', 'longitude', 'city', 'county',
        'acct_suspd_date', 'cust_orig_date', 'date_of_birth',
        'address_id'
    ]

    df = df.drop(columns=columns_to_drop, errors='ignore')

    return original_shape, df


def main():
    """Main execution function."""
    # Load data
    print("Loading data from data/raw/autoinsurance_churn.csv...")
    df = pd.read_csv('data/raw/autoinsurance_churn.csv')

    # Engineer features
    original_shape, df_engineered = engineer_features(df)

    # Print statistics
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)

    print(f"\nOriginal shape: {original_shape}")
    print(f"New shape: {df_engineered.shape}")
    print(f"Columns added: {df_engineered.shape[1] - original_shape[1] + 8}")  # +8 dropped cols

    # New columns (engineered ones)
    new_columns = [
        'tenure_years', 'tenure_bucket', 'is_new_customer',
        'income_midpoint', 'hmv_midpoint', 'premium_to_income_ratio',
        'premium_per_tenure_year', 'wealth_index', 'is_suspended',
        'risk_score', 'risk_score_clean', 'age_bucket', 'life_stage', 'stability_score',
        'state_churn_rate', 'simple_clv'
    ]

    print("\n[NULL COUNTS FOR NEW COLUMNS]")
    null_summary = df_engineered[new_columns].isnull().sum()
    null_pct = (null_summary / len(df_engineered) * 100).round(2)
    null_df = pd.DataFrame({
        'Null Count': null_summary,
        'Null %': null_pct
    })
    print(null_df[null_df['Null Count'] > 0])

    print("\n[TENURE_BUCKET DISTRIBUTION]")
    print(df_engineered['tenure_bucket'].value_counts().sort_index())

    print("\n[AGE_BUCKET DISTRIBUTION]")
    print(df_engineered['age_bucket'].value_counts().sort_index())

    print("\n[LIFE_STAGE DISTRIBUTION]")
    print(df_engineered['life_stage'].value_counts().head(10))

    print("\n[TOP 10 STATES BY CHURN RATE]")
    state_churn = df_engineered.groupby('state')['state_churn_rate'].first().sort_values(ascending=False)
    for state, rate in state_churn.head(10).items():
        print(f"{state}: {rate:.4f}")

    # Save engineered features
    output_path = 'data/features/engineered_features.csv'
    print(f"\nSaving engineered features to {output_path}...")
    df_engineered.to_csv(output_path, index=False)
    print("Done!")

    # Save model-ready features (drop leakage columns)
    # Note: acct_suspd_date already dropped in engineer_features()
    leakage_columns = ['is_suspended', 'risk_score']
    df_model_ready = df_engineered.drop(columns=leakage_columns, errors='ignore')

    model_ready_path = 'data/features/model_ready_features.csv'
    print(f"\nSaving model-ready features to {model_ready_path}...")
    df_model_ready.to_csv(model_ready_path, index=False)

    print("Leakage columns documented. Model-ready dataset saved to")
    print("data/features/model_ready_features.csv")
    print(f"Model-ready shape: {df_model_ready.shape}")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
