import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib
import shap
import numpy as np
import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import retention advisor with error handling
try:
    from src.advisor.retention_advisor import generate_retention_brief
    RETENTION_ADVISOR_AVAILABLE = True
except ImportError as e:
    RETENTION_ADVISOR_AVAILABLE = False
    RETENTION_ADVISOR_ERROR = str(e)

# Page configuration
st.set_page_config(
    page_title="Insurance Retention Analytics",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for professional blue/teal theme
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --success-color: #2ca02c;
        --accent-color: #17becf;
    }

    /* Clean typography */
    .main {
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    }

    /* Header styling */
    h1 {
        color: #1f77b4;
        font-weight: 600;
        padding-bottom: 1rem;
        border-bottom: 3px solid #17becf;
    }

    h2 {
        color: #2c5f7a;
        font-weight: 500;
        margin-top: 2rem;
    }

    h3 {
        color: #3a7ca5;
        font-weight: 500;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        color: #1f77b4;
    }

    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #5a5a5a;
        font-weight: 500;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }

    /* Sidebar styling - stronger selectors */
    section[data-testid="stSidebar"] {
        background-color: #f0f5f9 !important;
    }

    section[data-testid="stSidebar"] > div {
        background-color: #f0f5f9 !important;
    }

    section[data-testid="stSidebar"] .css-1d391kg {
        background-color: #f0f5f9 !important;
    }

    /* Sidebar text and titles */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #262730 !important;
    }

    /* Radio button labels */
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #262730 !important;
        font-weight: 500;
    }

    /* Radio button styling */
    section[data-testid="stSidebar"] .stRadio > label {
        color: #262730 !important;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        color: #262730 !important;
        font-weight: 500;
    }

    /* Sidebar radio selected state */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
        background-color: #1f77b4 !important;
    }

    /* Button styling */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }

    .stButton>button:hover {
        background-color: #17becf;
    }

    /* Card-like containers */
    div.block-container {
        padding-top: 2rem;
    }

    /* Clean metric cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Feature data initialization
def ensure_features_exist():
    """Generate model_ready_features.csv if it doesn't exist"""
    feature_path = "data/features/model_ready_features.csv"
    if not os.path.exists(feature_path):
        st.warning("⏳ Generating feature data for first-time setup... This may take 2-3 minutes.")
        try:
            # Create directories if they don't exist
            os.makedirs("data/features", exist_ok=True)
            # Run feature engineering script
            subprocess.run(["python", "src/features/engineer_features.py"], check=True)
            st.success("✅ Features generated!")
        except Exception as e:
            st.error(f"❌ Failed to generate features: {e}")
            st.stop()

# Call this before loading any data
if 'features_initialized' not in st.session_state:
    ensure_features_exist()
    st.session_state['features_initialized'] = True

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Portfolio Overview", "Customer Analysis"]
)

# Page 1: Portfolio Overview
if page == "Portfolio Overview":
    st.title("📊 Customer Portfolio Health Dashboard")
    st.markdown("### Key Performance Indicators")

    # Load customer segments data
    data_path = "outputs/analysis/customer_segments.csv"

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

        # Calculate metrics
        total_customers = len(df)
        avg_clv = df['predicted_clv'].mean()
        avg_churn_rate = df['churn_prob'].mean() * 100
        champions_count = df[df['segment'].str.contains('Champions', case=False, na=False)].shape[0]
        champions_percentage = (champions_count / total_customers) * 100

        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Total Customers",
                value=f"{total_customers:,}",
                delta=None
            )

        with col2:
            st.metric(
                label="Average Predicted CLV",
                value=f"${avg_clv:,.2f}",
                delta=None
            )

        with col3:
            st.metric(
                label="Average Churn Rate",
                value=f"{avg_churn_rate:.1f}%",
                delta=None
            )

        with col4:
            st.metric(
                label="Champions Percentage",
                value=f"{champions_percentage:.1f}%",
                delta=None
            )

        st.markdown("---")

        # Customer Risk-Value Matrix Scatter Plot
        st.markdown("### Customer Risk-Value Matrix")

        # Sample 2000 customers for performance
        df_sample = df.sample(min(2000, len(df)), random_state=42)

        color_map = {
            'Champions': '#2ca02c',
            'Stable': '#1f77b4',
            'At-Risk': '#ff7f0e'
        }

        fig = px.scatter(
            df_sample,
            x='churn_prob',
            y='predicted_clv',
            color='segment',
            size='actual_clv',
            color_discrete_map=color_map,
            title="Customer Risk-Value Matrix: The Churn-CLV Inverse Relationship",
            labels={
                'churn_prob': 'Churn Probability',
                'predicted_clv': 'Predicted CLV ($)'
            },
            opacity=0.6,
            size_max=20,
            hover_data=['individual_id', 'actual_clv']
        )

        # Update size range
        fig.update_traces(marker=dict(sizemin=5))

        # Add threshold lines
        fig.add_hline(y=8000, line_dash="dash", line_color="gray", annotation_text="High CLV Threshold ($8,000)")
        fig.add_vline(x=0.70, line_dash="dash", line_color="gray", annotation_text="High Churn Threshold (70%)")

        # Add annotation in top-right quadrant
        fig.add_annotation(
            x=0.85,
            y=15000,
            text="Zero customers in this quadrant<br>(High Churn + High CLV)",
            showarrow=False,
            font=dict(size=12, color="red"),
            align="center",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1
        )

        # Format axes
        fig.update_xaxes(tickformat='.0%')
        fig.update_yaxes(tickformat='$,.0f')

        # Update layout
        fig.update_layout(
            height=600,
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Key Business Insight
        st.info(
            "💡 **Key Insight:** Zero customers have both high churn risk (≥70%) and high CLV (≥$8,000). "
            "High-value customers are stable with only 7% churn rate. Strategic focus should be on "
            "first-year customer onboarding to convert new customers into long-term Champions."
        )

        # Top 20 At-Risk Customers
        with st.expander("📋 Top 20 At-Risk Customers"):
            at_risk_df = df[df['churn_prob'] >= 0.70].copy()
            top_at_risk = at_risk_df.sort_values('predicted_clv', ascending=False).head(20)

            if len(top_at_risk) > 0:
                # Format the churn probability as percentage
                top_at_risk_display = top_at_risk.copy()
                top_at_risk_display['churn_prob'] = top_at_risk_display['churn_prob'] * 100

                st.dataframe(
                    top_at_risk_display[['individual_id', 'churn_prob', 'predicted_clv', 'actual_clv', 'segment']],
                    use_container_width=True,
                    column_config={
                        "individual_id": "Customer ID",
                        "churn_prob": st.column_config.NumberColumn(
                            "Churn Probability",
                            format="%.1f%%",
                            help="Predicted probability of customer churn"
                        ),
                        "predicted_clv": st.column_config.NumberColumn(
                            "Predicted CLV",
                            format="$%.2f",
                            help="Predicted Customer Lifetime Value"
                        ),
                        "actual_clv": st.column_config.NumberColumn(
                            "Actual CLV",
                            format="$%.2f",
                            help="Actual Customer Lifetime Value"
                        ),
                        "segment": "Segment"
                    },
                    hide_index=True
                )
            else:
                st.success("✅ No high-risk customers (churn probability ≥ 70%) found!")

        st.markdown("---")

    else:
        st.error(f"Data file not found: {data_path}")
        st.info("Please run the customer segmentation analysis first.")

# Page 2: Customer Analysis
elif page == "Customer Analysis":
    st.title("🔍 Customer Risk Analysis")

    # Customer ID input
    customer_id = st.text_input("Enter Customer ID", placeholder="e.g., 221301989954")

    # Load model and feature names ONCE (outside customer-specific logic)
    if 'churn_model' not in st.session_state:
        try:
            # Load model
            model = joblib.load("outputs/models/best_churn_model.pkl")
            st.session_state['churn_model'] = model

            # Recreate feature names with EXACT same preprocessing as training
            # Load a sample to get the full column structure
            mapping_df = pd.read_csv("data/features/model_ready_features.csv", nrows=10000)

            # Step 1: Use EXACT categorical columns from training
            # (Hardcoded to match train_churn.py output)
            categorical_cols = ['state', 'marital_status', 'home_market_value',
                               'tenure_bucket', 'age_bucket', 'life_stage', 'income']

            # Filter to only columns that exist in the data
            categorical_cols = [col for col in categorical_cols if col in mapping_df.columns]

            # Step 2: One-hot encode with drop_first=True (EXACT same as training)
            mapping_encoded = pd.get_dummies(
                mapping_df,
                columns=categorical_cols,
                drop_first=True
            )

            # Step 3: Drop individual_id first
            mapping_encoded = mapping_encoded.drop(columns=['individual_id'], errors='ignore')

            # Step 4: Drop redundant raw features
            mapping_encoded = mapping_encoded.drop(
                columns=['days_tenure', 'age_in_years', 'curr_ann_amt'],
                errors='ignore'
            )

            # Step 5: Drop leaky/redundant features
            mapping_encoded = mapping_encoded.drop(
                columns=['state_churn_rate', 'tenure_years', 'simple_clv', 'premium_per_tenure_year'],
                errors='ignore'
            )

            # Step 6: Drop target column 'Churn' to get final feature set
            mapping_encoded = mapping_encoded.drop(columns=['Churn'], errors='ignore')

            # Store the final feature names (should be 72 features)
            st.session_state['real_feature_names'] = mapping_encoded.columns.tolist()

        except Exception as e:
            st.session_state['churn_model'] = None
            st.session_state['real_feature_names'] = []
            st.error(f"Error loading model: {e}")

    # Analyze button - only handles loading data into session_state
    if st.button("🔍 Analyze Customer", type="primary"):
        if customer_id:
            # Load customer segments data
            data_path = "outputs/analysis/customer_segments.csv"

            if os.path.exists(data_path):
                df = pd.read_csv(data_path)

                # Filter for the specific customer (handle float data type)
                try:
                    customer_row = df[df['individual_id'] == float(customer_id)]
                except ValueError:
                    st.error(f"❌ Invalid Customer ID format. Please enter a numeric ID.")
                    customer_row = pd.DataFrame()  # Empty dataframe

                if len(customer_row) > 0:
                    customer_data = customer_row.iloc[0]

                    # Store customer data in session_state for persistence across reruns
                    st.session_state['current_customer'] = {
                        'customer_id': str(customer_id),
                        'churn_prob': customer_data['churn_prob'],
                        'predicted_clv': customer_data['predicted_clv'],
                        'actual_clv': customer_data['actual_clv'],
                        'segment': customer_data['segment']
                    }
                else:
                    st.error(f"❌ Customer ID '{customer_id}' not found. Please check the ID and try again.")
            else:
                st.error(f"Data file not found: {data_path}")
                st.info("Please run the customer segmentation analysis first.")
        else:
            st.warning("⚠️ Please enter a Customer ID to analyze.")

    # Display customer analysis if data exists in session_state
    if 'current_customer' in st.session_state:
        customer = st.session_state['current_customer']
        display_customer_id = customer['customer_id']

        st.markdown("---")
        st.markdown(f"### Customer Profile: {display_customer_id}")

        # Display in 4 columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Churn Probability with color-coded background
            churn_prob = customer['churn_prob']
            churn_prob_pct = churn_prob * 100

            if churn_prob >= 0.70:
                st.markdown(
                    f"""
                    <div style="background-color: #ffcccc; padding: 20px; border-radius: 10px; border: 2px solid #ff0000;">
                        <h4 style="color: #cc0000; margin: 0;">⚠️ High Risk</h4>
                        <h2 style="color: #cc0000; margin: 10px 0;">{churn_prob_pct:.1f}%</h2>
                        <p style="color: #666; margin: 0;">Churn Probability</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif churn_prob >= 0.40:
                st.markdown(
                    f"""
                    <div style="background-color: #fff4cc; padding: 20px; border-radius: 10px; border: 2px solid #ffaa00;">
                        <h4 style="color: #cc8800; margin: 0;">⚡ Medium Risk</h4>
                        <h2 style="color: #cc8800; margin: 10px 0;">{churn_prob_pct:.1f}%</h2>
                        <p style="color: #666; margin: 0;">Churn Probability</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color: #ccffcc; padding: 20px; border-radius: 10px; border: 2px solid #00aa00;">
                        <h4 style="color: #008800; margin: 0;">✅ Low Risk</h4>
                        <h2 style="color: #008800; margin: 10px 0;">{churn_prob_pct:.1f}%</h2>
                        <p style="color: #666; margin: 0;">Churn Probability</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col2:
            # Predicted CLV
            predicted_clv = customer['predicted_clv']
            st.metric(
                label="Predicted CLV",
                value=f"${predicted_clv:,.2f}"
            )

        with col3:
            # Actual CLV with delta
            actual_clv = customer['actual_clv']
            clv_diff = actual_clv - predicted_clv

            st.metric(
                label="Actual CLV",
                value=f"${actual_clv:,.2f}",
                delta=f"${clv_diff:,.2f} vs Predicted"
            )

        with col4:
            # Segment badge
            segment = customer['segment']

            st.markdown("#### Customer Segment")

            if 'Champions' in segment:
                st.success(f"✅ {segment}")
            elif 'At-Risk' in segment:
                st.error(f"⚠️ {segment}")
            elif 'Stable' in segment:
                st.info(f"ℹ️ {segment}")
            else:
                st.warning(f"📊 {segment}")

        st.markdown("---")
        st.subheader("📊 Why This Customer Might Churn (SHAP Analysis)")

        with st.spinner("Computing SHAP values..."):
            try:
                # Get model from session state
                model = st.session_state.get('churn_model')
                if model is None:
                    st.warning("⚠️ Model not loaded. Please refresh the page.")
                    st.stop()

                # Load customer features
                features_df = pd.read_csv("data/features/model_ready_features.csv")

                # Check if customer exists
                if float(display_customer_id) not in features_df['individual_id'].values:
                    st.warning("⚠️ Customer features not found in the model-ready dataset.")
                else:
                    # Get customer index and filter
                    customer_idx = features_df[features_df['individual_id'] == float(display_customer_id)].index[0]
                    customer_row = features_df.loc[[customer_idx]].copy()

                # Apply IDENTICAL preprocessing as training
                # Step 1: Use EXACT categorical columns from training (hardcoded)
                categorical_cols = ['state', 'marital_status', 'home_market_value',
                                   'tenure_bucket', 'age_bucket', 'life_stage', 'income']

                # Filter to only columns that exist in customer data
                categorical_cols = [col for col in categorical_cols if col in customer_row.columns]

                # Step 2: One-hot encode with drop_first=True (EXACT same as training)
                customer_row = pd.get_dummies(customer_row, columns=categorical_cols, drop_first=True)

                # Step 3: Drop individual_id
                customer_row = customer_row.drop(columns=['individual_id'], errors='ignore')

                # Step 4: Drop redundant raw features
                customer_row = customer_row.drop(
                    columns=['days_tenure', 'age_in_years', 'curr_ann_amt'],
                    errors='ignore'
                )

                # Step 5: Drop leaky/redundant features
                customer_row = customer_row.drop(
                    columns=['state_churn_rate', 'tenure_years', 'simple_clv', 'premium_per_tenure_year'],
                    errors='ignore'
                )

                # Step 6: Drop target column
                customer_row = customer_row.drop(columns=['Churn'], errors='ignore')

                # Step 7: Align with expected feature names (should be 72)
                expected_features = st.session_state.get('real_feature_names', [])

                # Add missing columns with 0 (for one-hot encoded categories not present)
                for col in expected_features:
                    if col not in customer_row.columns:
                        customer_row[col] = 0

                # Reorder to match exactly (CRITICAL for model prediction)
                customer_features_encoded = customer_row[expected_features]

                # Compute SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(customer_features_encoded)

                # Extract SHAP values for churn class (flatten to 1D array)
                if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
                    shap_values_array = shap_values.values[0, :, 1]  # Binary classification, churn class
                elif hasattr(shap_values, 'values') and len(shap_values.values.shape) == 2:
                    shap_values_array = shap_values.values[0, :]
                else:
                    shap_values_array = shap_values.values[0]

                # Get real feature names from session state (recreated from training pipeline)
                real_feature_names = st.session_state.get('real_feature_names', [])

                # Validate lengths match (should both be 72)
                if len(real_feature_names) != len(shap_values_array):
                    st.error(
                        f"⚠️ Feature mismatch: {len(real_feature_names)} names vs "
                        f"{len(shap_values_array)} SHAP values. Using fallback names."
                    )
                    real_feature_names = [f"feature_{i}" for i in range(len(shap_values_array))]

                # Create DataFrame with real feature names and SHAP values
                shap_df = pd.DataFrame({
                    'feature': real_feature_names,
                    'shap_value': shap_values_array
                })

                # Get top 5 features by absolute SHAP value
                shap_df['abs_shap'] = shap_df['shap_value'].abs()
                top_5_shap = shap_df.nlargest(5, 'abs_shap').sort_values('shap_value', ascending=True)

                # Store top 3 SHAP features in session_state for AI advisor
                top_3_features = shap_df.nlargest(3, 'abs_shap')
                st.session_state['shap_features'] = top_3_features[['feature', 'shap_value']].values.tolist()

                # Create plotly horizontal bar chart
                colors = ['#2ca02c' if val < 0 else '#d62728' for val in top_5_shap['shap_value'].values]

                fig_shap = px.bar(
                    top_5_shap,
                    x='shap_value',
                    y='feature',
                    orientation='h',
                    title='Top 5 Features Influencing Churn Risk',
                    labels={
                        'shap_value': 'SHAP Value (Impact on Churn)',
                        'feature': 'Feature Name'
                    },
                    hover_data={'feature': True, 'shap_value': ':.4f'}
                )

                fig_shap.update_traces(marker_color=colors)
                fig_shap.add_vline(x=0, line_dash="dash", line_color="gray")
                fig_shap.update_layout(height=400, showlegend=False)

                st.plotly_chart(fig_shap, use_container_width=True)

                st.info(
                    "📊 **How to read this chart:** "
                    "🔴 Red bars (positive values) increase churn risk. "
                    "🟢 Green bars (negative values) decrease churn risk. "
                    "Longer bars indicate stronger impact."
                )

            except FileNotFoundError as e:
                st.warning(f"⚠️ SHAP analysis unavailable. Model or feature data may be missing: {e}")
            except Exception as e:
                st.warning(f"⚠️ SHAP analysis unavailable. Error: {e}")

    # AI Retention Advisor Integration
    if 'current_customer' in st.session_state and 'shap_features' in st.session_state:
        st.markdown("---")
        st.subheader("🤖 AI-Powered Retention Strategy")

        st.info("💡 Click below to get AI-powered retention recommendations based on this customer's risk profile")

        if st.button("Generate AI Retention Brief", type="primary", use_container_width=True):
            if not RETENTION_ADVISOR_AVAILABLE:
                st.error(f"⚠️ Retention advisor module not available: {RETENTION_ADVISOR_ERROR}")
                st.info("Please ensure advisor/retention_advisor.py exists and all dependencies are installed.")
            else:
                try:
                    # Get customer data from session_state
                    customer = st.session_state['current_customer']

                    # Get top 3 SHAP features from session_state
                    top_3_shap = [(f, v) for f, v in st.session_state['shap_features']]

                    # Generate retention brief with AI
                    with st.spinner("Generating retention strategy with Gemini AI..."):
                        brief = generate_retention_brief(
                            customer['customer_id'],
                            customer['churn_prob'],
                            customer['predicted_clv'],
                            customer['actual_clv'],
                            top_3_shap
                        )

                    # Display the brief
                    with st.expander("📋 AI-Generated Retention Brief", expanded=True):
                        st.markdown(brief)

                except Exception as e:
                    st.error(f"⚠️ Failed to generate brief: {str(e)}")
                    st.info("Check your Gemini API key and internet connection.")
