"""
Fairness and Bias Analysis for Credit Scoring
Detects disparate impact across protected attributes.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, recall_score
from pathlib import Path


def load_test_data():
    """Load test dataset with original features"""
    df = pd.read_csv("data/uci/UCI_Credit_Card.csv")
    df = df.rename(columns={'default.payment.next.month': 'default'})
    return df


def get_group_metrics(y_true, y_pred, probabilities, threshold=0.20):
    """
    Calculate fairness metrics for a group.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (at threshold)
        probabilities: Predicted probabilities
        threshold: Decision threshold
        
    Returns:
        Dict with fairness metrics
    """
    # Confusion matrix (force 2x2 even if one class is missing)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Handle edge cases where group might have only one class
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Shouldn't happen with labels parameter, but safe fallback
        tn = fp = fn = tp = 0
    
    # Metrics
    recall = recall_score(y_true, y_pred, zero_division=0)  # Catch defaulters
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # Miss defaulters
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False alarm
    
    # Approval rate (predict non-default)
    approval_rate = float((y_pred == 0).sum() / len(y_pred)) if len(y_pred) > 0 else 0.0
    
    # Average probability
    avg_prob = float(probabilities.mean())
    
    return {
        "sample_size": len(y_true),
        "recall": round(recall, 4),
        "false_negative_rate": round(fnr, 4),
        "false_positive_rate": round(fpr, 4),
        "approval_rate": round(approval_rate, 4),
        "avg_default_probability": round(avg_prob, 4),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn)
    }


def analyze_fairness_by_attribute(attribute: str, threshold: float = 0.20):
    """
    Analyze fairness for a protected attribute.
    
    Args:
        attribute: Column name (SEX, EDUCATION, MARRIAGE)
        threshold: Decision threshold
        
    Returns:
        Dict with metrics per group and disparities
    """
    # Load data
    df = load_test_data()
    
    # Load model and make predictions on full dataset
    model = joblib.load("src/uci_model_calibrated.pkl")
    feature_names = joblib.load("src/feature_names.pkl")
    
    # Engineer features in batch (MUCH faster than row-by-row)
    from src.model import engineer_features_batch
    
    print(f"  Engineering features for {len(df)} samples...")
    features = engineer_features_batch(df)
    
    print(f"  Making predictions...")
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'true': df['default'].values,
        'pred': predictions,
        'prob': probabilities,
        attribute: df[attribute].values
    })
    
    # Get unique groups
    groups = sorted(results_df[attribute].unique())
    
    # Calculate metrics per group
    group_metrics = {}
    for group in groups:
        mask = results_df[attribute] == group
        group_data = results_df[mask]
        
        metrics = get_group_metrics(
            y_true=group_data['true'].values,
            y_pred=group_data['pred'].values,
            probabilities=group_data['prob'].values,
            threshold=threshold
        )
        # Convert numpy int64 to native Python int for JSON serialization
        group_metrics[int(group)] = metrics
    
    # Calculate disparities (compare each group to reference group)
    reference_group = int(groups[0])
    disparities = {}
    max_approval_diff = 0.0
    
    for group in groups:
        group = int(group)
        if group == reference_group:
            continue
            
        ref_metrics = group_metrics[reference_group]
        grp_metrics = group_metrics[group]
        
        approval_diff_pct = abs((grp_metrics['approval_rate'] - ref_metrics['approval_rate']) * 100)
        max_approval_diff = max(max_approval_diff, approval_diff_pct)
        
        disparities[f"{reference_group}_vs_{group}"] = {
            "recall_diff": round(grp_metrics['recall'] - ref_metrics['recall'], 4),
            "recall_diff_pct": round((grp_metrics['recall'] - ref_metrics['recall']) * 100, 2),
            "fnr_diff": round(grp_metrics['false_negative_rate'] - ref_metrics['false_negative_rate'], 4),
            "approval_rate_diff": round(grp_metrics['approval_rate'] - ref_metrics['approval_rate'], 4),
            "approval_rate_diff_pct": round((grp_metrics['approval_rate'] - ref_metrics['approval_rate']) * 100, 2)
        }
    
    # Add summary metrics
    if max_approval_diff > 15:
        severity = "HIGH"
    elif max_approval_diff > 10:
        severity = "MODERATE"
    else:
        severity = "ACCEPTABLE"
    
    disparities_summary = {
        "max_approval_rate_diff": round(max_approval_diff, 2),
        "severity": severity,
        "comparisons": disparities
    }
    
    return {
        "attribute": attribute,
        "threshold": float(threshold),
        "groups": group_metrics,
        "disparities": disparities_summary
    }


def detect_bias(threshold_pct: float = 10.0):
    """
    Run full bias detection across all protected attributes.
    
    Args:
        threshold_pct: Flag bias if difference > this percentage (default 10%)
        
    Returns:
        Dict with analysis for SEX, EDUCATION, MARRIAGE
    """
    protected_attributes = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    results = {}
    bias_flags = []
    
    for attr in protected_attributes:
        print(f"\nAnalyzing {attr}...")
        analysis = analyze_fairness_by_attribute(attr)
        results[attr] = analysis
        
        # Check for bias
        for comparison, metrics in analysis['disparities']['comparisons'].items():
            # Flag if approval rate differs by > threshold
            if abs(metrics['approval_rate_diff_pct']) > threshold_pct:
                bias_flags.append({
                    "attribute": attr,
                    "comparison": comparison,
                    "metric": "approval_rate",
                    "difference_pct": metrics['approval_rate_diff_pct'],
                    "severity": "HIGH" if abs(metrics['approval_rate_diff_pct']) > 15 else "MODERATE"
                })
            
            # Flag if FNR differs significantly
            if abs(metrics['fnr_diff']) > threshold_pct / 100:
                bias_flags.append({
                    "attribute": attr,
                    "comparison": comparison,
                    "metric": "false_negative_rate",
                    "difference": metrics['fnr_diff'],
                    "severity": "HIGH" if abs(metrics['fnr_diff']) > 0.15 else "MODERATE"
                })
    
    results['bias_flags'] = bias_flags
    results['bias_detected'] = len(bias_flags) > 0
    
    return results


def print_fairness_report():
    """Generate human-readable fairness report"""
    print("="*60)
    print("FAIRNESS & BIAS ANALYSIS")
    print("="*60)
    
    results = detect_bias(threshold_pct=10.0)
    
    # Print per-attribute analysis
    for attr in ['SEX', 'EDUCATION', 'MARRIAGE']:
        analysis = results[attr]
        
        print(f"\n{'-'*60}")
        print(f"Protected Attribute: {attr}")
        print(f"{'-'*60}")
        
        # Print group metrics
        for group, metrics in analysis['groups'].items():
            print(f"\n{attr} = {group} (n={metrics['sample_size']})")
            print(f"  Recall (catch defaulters):     {metrics['recall']:.4f}")
            print(f"  False Negative Rate:           {metrics['false_negative_rate']:.4f}")
            print(f"  Approval Rate:                 {metrics['approval_rate']:.4f} ({metrics['approval_rate']*100:.1f}%)")
            print(f"  Avg Default Probability:       {metrics['avg_default_probability']:.4f}")
        
        # Print disparities
        if analysis['disparities']['comparisons']:
            print(f"\n  Disparities:")
            for comparison, disp in analysis['disparities']['comparisons'].items():
                print(f"    {comparison}:")
                print(f"      Recall difference:        {disp['recall_diff']:+.4f} ({disp['recall_diff_pct']:+.2f}%)")
                print(f"      Approval rate difference: {disp['approval_rate_diff']:+.4f} ({disp['approval_rate_diff_pct']:+.2f}%)")
    
    # Print bias flags
    print(f"\n{'='*60}")
    print("BIAS DETECTION SUMMARY")
    print(f"{'='*60}")
    
    if results['bias_detected']:
        print(f"\n⚠️  {len(results['bias_flags'])} potential bias issues detected:\n")
        for flag in results['bias_flags']:
            print(f"  [{flag['severity']}] {flag['attribute']} - {flag['comparison']}")
            print(f"    Metric: {flag['metric']}")
            if 'difference_pct' in flag:
                print(f"    Difference: {flag['difference_pct']:+.2f}%\n")
            else:
                print(f"    Difference: {flag['difference']:+.4f}\n")
    else:
        print("\n✓ No significant bias detected (threshold: 10%)")
    
    print(f"{'='*60}")
    print("Note: Fairness in ML is complex. These metrics show")
    print("disparate impact but don't prove discrimination.")
    print("Consult with legal/ethics teams for production use.")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    print_fairness_report()
