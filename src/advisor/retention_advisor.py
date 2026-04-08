from .gemini_client import GeminiClient


def generate_retention_brief(
    customer_id: str,
    churn_prob: float,
    predicted_clv: float,
    actual_clv: float,
    top_shap_features: list
) -> str:
    """
    Generate a personalized retention brief for an at-risk customer.

    Args:
        customer_id: Customer identifier
        churn_prob: Predicted churn probability (0-1)
        predicted_clv: Predicted customer lifetime value in dollars
        actual_clv: Actual customer lifetime value in dollars
        top_shap_features: List of (feature_name, shap_value) tuples

    Returns:
        AI-generated retention brief with structured recommendations
    """
    # Create Gemini client
    client = GeminiClient()

    # Format SHAP features for the prompt
    shap_explanations = []
    for i, (feature, shap_value) in enumerate(top_shap_features[:3], 1):
        direction = "increasing" if shap_value > 0 else "decreasing"
        shap_explanations.append(
            f"   {i}. {feature}: SHAP value = {shap_value:.2f} ({direction} churn risk)"
        )

    shap_features_text = "\n".join(shap_explanations)

    # Construct detailed prompt
    prompt = f"""You are an expert insurance customer retention strategist. Generate a comprehensive retention brief for this at-risk customer.

CUSTOMER PROFILE:
- Customer ID: {customer_id}
- Churn Probability: {churn_prob:.1%}
- Predicted Lifetime Value: ${predicted_clv:,.2f}
- Actual Lifetime Value: ${actual_clv:,.2f}

TOP FACTORS DRIVING CHURN RISK (SHAP Analysis):
{shap_features_text}

Please provide a structured retention brief with the following sections:

1. RISK LEVEL ASSESSMENT
   Categorize as Critical/High/Medium/Low and explain why based on churn probability and CLV.

2. ROOT CAUSE ANALYSIS
   Explain the top 3 SHAP features in plain business English. What do they mean for this customer's behavior and why are they driving churn risk?

3. RECOMMENDED ACTIONS (3 specific tactics)
   For each action, provide:
   - Concrete tactic (what to do)
   - Expected outcome (impact on retention)
   - Implementation complexity (easy/medium/hard)

4. ROI CALCULATION
   - Estimated cost of retention intervention
   - CLV at risk if customer churns
   - Expected ROI if intervention succeeds

5. PRIORITY & TIMELINE
   When should we act (immediate/this week/this month) and what's the sequence of actions?

Format your response clearly with headers and bullet points. Be specific and actionable."""

    # Generate brief using Gemini
    try:
        brief = client.generate_brief(prompt)
        return brief
    except Exception as e:
        return f"Error generating retention brief: {str(e)}"
