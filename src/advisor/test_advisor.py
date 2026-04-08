"""
Test script for the AI-powered retention advisor system.
Tests with a real high-risk customer profile.
"""

from src.advisor.retention_advisor import generate_retention_brief


def main():
    print("=" * 80)
    print("AI-POWERED RETENTION ADVISOR TEST")
    print("=" * 80)

    # Test customer data (high churn risk, relatively low CLV)
    customer_id = "221301989954"
    churn_prob = 0.8228
    predicted_clv = 1678.52
    actual_clv = 664.73

    # Top SHAP features driving churn prediction
    top_shap_features = [
        ("tenure_bucket_<1yr", -7143.04),
        ("premium_to_income_ratio", -2200.00),
        ("is_new_customer", -498.52)
    ]

    print(f"\nGenerating retention brief for Customer {customer_id}...")
    print(f"Churn Risk: {churn_prob:.1%}")
    print(f"Predicted CLV: ${predicted_clv:,.2f}")
    print(f"Actual CLV: ${actual_clv:,.2f}")
    print("\nCalling Gemini 2.5 Flash API...\n")
    print("=" * 80)

    # Generate retention brief
    brief = generate_retention_brief(
        customer_id=customer_id,
        churn_prob=churn_prob,
        predicted_clv=predicted_clv,
        actual_clv=actual_clv,
        top_shap_features=top_shap_features
    )

    # Print the generated brief
    print(brief)
    print("\n" + "=" * 80)
    print("RETENTION BRIEF GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
