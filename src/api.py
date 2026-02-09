"""
FastAPI Credit Scoring Service with Explainability
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.schema import CreditRequest
from src.model import predict_default, engineer_features
from src.predict import make_decision, get_decision_reasons, optimal_threshold
from src.explain import get_explanation
import time
from datetime import datetime

app = FastAPI(
    title="Explainable Credit Risk Engine (XCRE)",
    description="Production-grade credit scoring API with cost-optimized decisions",
    version="1.0.0"
)

# Enable CORS for React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "service": "Explainable Credit Risk Engine",
        "status": "running",
        "version": "1.0.0",
        "optimal_threshold": round(optimal_threshold, 3)
    }


@app.post("/score")
def score_credit(req: CreditRequest):
    """
    Score a credit application and return decision.
    
    Returns:
    - default_probability: Calibrated probability of default (0-1)
    - credit_score: FICO-style score (300-850)
    - decision: APPROVE, MANUAL_REVIEW, or REJECT
    - risk_level: HIGH, MEDIUM, or LOW
    - threshold: Decision threshold used
    """
    try:
        start_time = time.time()
        
        # Convert Pydantic model to dict for processing
        data = req.model_dump()
        
        # Engineer features (needed for SHAP)
        engineered_features = engineer_features(data)
        
        # Get calibrated probability
        prob = predict_default(data)
        
        # Make decision
        decision = make_decision(prob)
        
        # Get SHAP explanations (top 5 reasons)
        reasons = get_explanation(engineered_features, top_n=5)
        
        # Convert probability to credit score (FICO-style: 300-850)
        # Lower probability = higher score
        credit_score = int(850 - (prob * 550))
        credit_score = max(300, min(850, credit_score))  # Clamp to valid range
        
        # Get detailed decision reasoning
        decision_details = get_decision_reasons(prob)
        
        # Calculate processing time
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        # Log prediction for monitoring (optional - don't break API if fails)
        try:
            from src.monitor import log_prediction
            max_delay_value = float(engineered_features['max_delay'].iloc[0])
            log_prediction(probability=prob, decision=decision, max_delay=max_delay_value)
        except:
            pass  # Monitoring is optional
        
        return {
            "default_probability": round(prob, 4),
            "credit_score": credit_score,
            "decision": decision,
            "risk_level": decision_details["risk_level"],
            "confidence": decision_details["confidence"],
            "threshold_used": round(optimal_threshold, 3),
            "reasons": reasons,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_ms": processing_time_ms,
                "model_version": "1.0.0"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/score/batch")
def score_batch(requests: list[CreditRequest]):
    """
    Score multiple credit applications in batch.
    
    Useful for processing historical data or bulk evaluations.
    """
    try:
        results = []
        for req in requests:
            data = req.model_dump()
            prob = predict_default(data)
            decision = make_decision(prob)
            credit_score = int(850 - (prob * 550))
            credit_score = max(300, min(850, credit_score))
            
            results.append({
                "default_probability": round(prob, 4),
                "credit_score": credit_score,
                "decision": decision
            })
        
        return {
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


@app.get("/model/info")
def model_info():
    """Get model configuration and statistics"""
    import joblib
    
    try:
        feature_names = joblib.load("src/feature_names.pkl")
        metrics = joblib.load("src/model_metrics.pkl")
        
        return {
            "model_type": "XGBoost with Probability Calibration",
            "num_features": len(feature_names),
            "optimal_threshold": round(optimal_threshold, 3),
            "performance": {
                "test_auc": round(metrics.get("test_auc", 0), 4),
                "test_recall": round(metrics.get("test_recall", 0), 4),
                "test_precision": round(metrics.get("test_precision", 0), 4),
                "cost_reduction_pct": round(metrics.get("cost_reduction_pct", 0), 2)
            },
            "cost_configuration": {
                "false_negative_cost": metrics.get("cost_fn", 5),
                "false_positive_cost": metrics.get("cost_fp", 1),
                "optimization": "Minimize total business cost"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load model info: {str(e)}"
        )


@app.get("/monitoring")
def monitoring_summary():
    """
    Get simple monitoring stats: approval rate and drift check.
    """
    try:
        from src.monitor import get_approval_rate, check_drift
        return {
            "approval_rate": get_approval_rate(),
            "drift_check": check_drift()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Monitoring error: {str(e)}"
        )


@app.get("/fairness")
def fairness_analysis(attribute: str = None):
    """
    Get fairness analysis for protected attributes.
    
    Args:
        attribute: Optional. Filter by SEX, EDUCATION, or MARRIAGE
        
    Returns:
        Fairness metrics and bias detection across groups
    """
    try:
        from src.fairness import analyze_fairness_by_attribute, detect_bias
        
        if attribute:
            # Single attribute analysis
            if attribute not in ['SEX', 'EDUCATION', 'MARRIAGE']:
                raise HTTPException(status_code=400, detail="Invalid attribute. Use SEX, EDUCATION, or MARRIAGE")
            return analyze_fairness_by_attribute(attribute)
        else:
            # Full bias detection
            return detect_bias(threshold_pct=10.0)
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fairness analysis error: {str(e)}"
        )
