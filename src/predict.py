"""
Decision engine with optimal threshold and business rules
"""
import joblib

# Load optimal threshold from training
try:
    optimal_threshold = joblib.load("src/optimal_threshold.pkl")
    print(f"✓ Loaded optimal threshold: {optimal_threshold:.3f}")
except Exception as e:
    print(f"Warning: Could not load optimal threshold, using default 0.20")
    optimal_threshold = 0.20


def make_decision(prob: float) -> str:
    """
    Make credit decision based on calibrated probability.
    
    Uses cost-optimized threshold from training (typically 0.20).
    
    Business Rules:
    - High risk (prob >= optimal_threshold): REJECT
    - Borderline: MANUAL_REVIEW (optional tier)  
    - Low risk: APPROVE
    
    Args:
        prob: Calibrated default probability (0-1)
        
    Returns:
        Decision string: "APPROVE", "MANUAL_REVIEW", or "REJECT"
    """
    # Primary decision based on optimal threshold
    if prob >= optimal_threshold:
        # High risk - automatic rejection
        # Could add manual review tier for borderline cases
        if prob >= 0.35:
            return "REJECT"
        else:
            return "MANUAL_REVIEW"
    else:
        # Low risk - approval
        return "APPROVE"


def get_decision_reasons(prob: float, threshold: float = None) -> dict:
    """
    Provide reasoning for the decision.
    
    Args:
        prob: Default probability
        threshold: Decision threshold (uses optimal if None)
        
    Returns:
        Dictionary with decision metadata
    """
    if threshold is None:
        threshold = optimal_threshold
    
    decision = make_decision(prob)
    
    return {
        "decision": decision,
        "probability": round(prob, 4),
        "threshold": round(threshold, 4),
        "risk_level": "HIGH" if prob >= 0.35 else "MEDIUM" if prob >= threshold else "LOW",
        "confidence": "High" if abs(prob - threshold) > 0.15 else "Medium" if abs(prob - threshold) > 0.05 else "Borderline"
    }
