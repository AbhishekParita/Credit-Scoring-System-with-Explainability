"""
Simple prediction logging for basic drift monitoring.
Minimal implementation - just logs to CSV.
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

# File paths
LOG_FILE = "monitoring/predictions.csv"
TRAINING_STATS_PATH = "src/training_stats.json"

def init_monitoring():
    """Create monitoring directory if needed"""
    Path("monitoring").mkdir(exist_ok=True)
    print(f"✓ Monitoring initialized: {LOG_FILE}")


def log_prediction(probability: float, decision: str, max_delay: Optional[float] = None):
    """
    Log a prediction to CSV file.
    
    Args:
        probability: Default probability [0-1]
        decision: APPROVE, MANUAL_REVIEW, or REJECT
        max_delay: Value of max_delay feature (optional)
    """
    # Create log entry
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'probability': probability,
        'decision': decision,
        'max_delay': max_delay
    }
    
    # Append to CSV
    df = pd.DataFrame([log_entry])
    
    if Path(LOG_FILE).exists():
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        Path("monitoring").mkdir(exist_ok=True)
        df.to_csv(LOG_FILE, mode='w', header=True, index=False)


def get_approval_rate() -> dict:
    """Calculate basic approval rate from logs"""
    if not Path(LOG_FILE).exists():
        return {"error": "No predictions logged yet"}
    
    df = pd.read_csv(LOG_FILE)
    
    total = len(df)
    approved = (df['decision'] == 'APPROVE').sum()
    rejected = (df['decision'] == 'REJECT').sum()
    
    return {
        "total_predictions": total,
        "approved": int(approved),
        "rejected": int(rejected),
        "approval_rate": round(approved / total * 100, 2) if total > 0 else 0,
        "avg_probability": round(df['probability'].mean(), 4) if total > 0 else 0
    }


def save_training_stats(max_delay_mean: float):
    """Save training baseline for drift comparison"""
    stats = {
        "max_delay_mean": max_delay_mean,
        "saved_at": datetime.now().isoformat()
    }
    
    with open(TRAINING_STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Training baseline saved: max_delay_mean={max_delay_mean:.4f}")


def check_drift() -> dict:
    """Simple drift check: compare production vs training mean"""
    if not Path(LOG_FILE).exists():
        return {"error": "No predictions logged"}
    
    if not Path(TRAINING_STATS_PATH).exists():
        return {"error": "Training stats not found"}
    
    # Load data
    df = pd.read_csv(LOG_FILE)
    with open(TRAINING_STATS_PATH) as f:
        train_stats = json.load(f)
    
    # Filter valid max_delay values
    df_filtered = df[df['max_delay'].notna()]
    
    if df_filtered.empty:
        return {"error": "No max_delay data"}
    
    # Compare means
    train_mean = train_stats['max_delay_mean']
    prod_mean = df_filtered['max_delay'].mean()
    diff_pct = ((prod_mean - train_mean) / train_mean * 100) if train_mean != 0 else 0
    
    return {
        "feature": "max_delay",
        "training_mean": round(train_mean, 2),
        "production_mean": round(prod_mean, 2),
        "difference_pct": round(diff_pct, 2),
        "samples": len(df_filtered)
    }


if __name__ == "__main__":
    init_monitoring()
    print("\nUsage:")
    print("  from src.monitor import log_prediction, get_approval_rate, check_drift")
    print("  log_prediction(probability=0.15, decision='APPROVE', max_delay=2.0)")

