"""
SHAP Explainability for Credit Scoring
Per-user explanations using TreeExplainer
"""
import shap
import joblib
import numpy as np
import pandas as pd

# Load base XGBoost model (SHAP works on tree models, not calibrated wrapper)
base_model = joblib.load("src/uci_model.pkl")
feature_names = joblib.load("src/feature_names.pkl")

# Initialize SHAP explainer with booster (fixes XGBoost compatibility issue)
# Use get_booster() to avoid base_score parsing error
explainer = shap.TreeExplainer(base_model.get_booster(), feature_names=feature_names, model_output="raw")
print(f"✓ SHAP TreeExplainer initialized with {len(feature_names)} features")


def get_explanation(engineered_features: pd.DataFrame, top_n: int = 5) -> list:
    """
    Get top N feature contributions for a prediction.
    
    Args:
        engineered_features: DataFrame with engineered features (24 features)
        top_n: Number of top contributors to return
        
    Returns:
        List of dicts with feature and impact (SHAP value)
    """
    # Compute SHAP values (returns array, not tuple when using booster)
    shap_values = explainer.shap_values(engineered_features)
    
    # If shap_values is 2D (1 sample, 24 features), get first row
    if len(shap_values.shape) == 2:
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values
    
    # Get feature contributions
    contributions = []
    for i, feature in enumerate(feature_names):
        contributions.append({
            "feature": feature,
            "impact": float(shap_vals[i])
        })
    
    # Sort by absolute impact (most influential first)
    contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
    
    # Return top N
    return contributions[:top_n]
