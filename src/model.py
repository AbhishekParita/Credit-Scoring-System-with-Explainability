"""
Model loading and feature engineering for credit scoring API
"""
import joblib
import numpy as np
import pandas as pd
from typing import Dict
from src.calibration import CalibratedClassifier  # Import for unpickling

# Load models and artifacts at startup
try:
    calibrated_model = joblib.load("src/uci_model_calibrated.pkl")
    feature_names = joblib.load("src/feature_names.pkl")
    print(f"✓ Loaded calibrated model with {len(feature_names)} features")
except Exception as e:
    print(f"Error loading models: {e}")
    raise


def engineer_features(data: Dict) -> pd.DataFrame:
    """
    Recreate the exact feature engineering from training pipeline.
    
    Args:
        data: Dictionary with raw UCI fields
        
    Returns:
        DataFrame with engineered features matching training schema
    """
    # Convert input to DataFrame for easier manipulation
    df = pd.DataFrame([data])
    
    # 1. Bill Amount Features (from BILL_AMT1-6)
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    df['avg_bill_amount'] = df[bill_cols].mean(axis=1)
    df['max_bill'] = df[bill_cols].max(axis=1)
    df['min_bill'] = df[bill_cols].min(axis=1)
    df['bill_volatility'] = df[bill_cols].std(axis=1)
    
    # Bill trend (slope over 6 months)
    for i, col in enumerate(bill_cols):
        df[f'bill_time_{i}'] = df[col] * (i + 1)
    df['bill_trend'] = df[[f'bill_time_{i}' for i in range(6)]].mean(axis=1) - df['avg_bill_amount']
    df.drop([f'bill_time_{i}' for i in range(6)], axis=1, inplace=True)
    
    # 2. Payment Amount Features (from PAY_AMT1-6)
    pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    df['total_paid'] = df[pay_cols].sum(axis=1)
    df['avg_payment'] = df[pay_cols].mean(axis=1)
    df['max_payment'] = df[pay_cols].max(axis=1)
    df['payment_volatility'] = df[pay_cols].std(axis=1)
    
    # Payment ratio (critical metric)
    df['payment_ratio'] = df['total_paid'] / (df[bill_cols].sum(axis=1) + 1)
    df['payment_to_income'] = df['total_paid'] / (df['LIMIT_BAL'] + 1)
    
    # 3. Payment Status Features (from PAY_0-6)
    pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    df['max_delay'] = df[pay_status_cols].max(axis=1)
    df['avg_delay'] = df[pay_status_cols].mean(axis=1)
    df['num_late_payments'] = (df[pay_status_cols] > 0).sum(axis=1)
    df['num_on_time_payments'] = (df[pay_status_cols] == 0).sum(axis=1)
    df['delay_volatility'] = df[pay_status_cols].std(axis=1)
    df['worst_payment_status'] = df[pay_status_cols].max(axis=1)
    
    # 4. Ratio and Interaction Features
    df['debt_to_income'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
    df['credit_utilization'] = df['avg_bill_amount'] / (df['LIMIT_BAL'] + 1)
    df['bill_to_income'] = df['max_bill'] / (df['LIMIT_BAL'] + 1)
    
    # Fill NaN values (from std calculations)
    df = df.fillna(0)
    
    # Select only the features used in training (in correct order)
    return df[feature_names]


def engineer_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch version of feature engineering for multiple rows at once.
    Much faster than calling engineer_features() in a loop.
    
    Args:
        df: DataFrame with raw UCI fields (multiple rows)
        
    Returns:
        DataFrame with engineered features matching training schema
    """
    # Work on a copy to avoid modifying original
    data = df.copy()
    
    # 1. Bill Amount Features (from BILL_AMT1-6)
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    data['avg_bill_amount'] = data[bill_cols].mean(axis=1)
    data['max_bill'] = data[bill_cols].max(axis=1)
    data['min_bill'] = data[bill_cols].min(axis=1)
    data['bill_volatility'] = data[bill_cols].std(axis=1)
    
    # Bill trend (slope over 6 months)
    for i, col in enumerate(bill_cols):
        data[f'bill_time_{i}'] = data[col] * (i + 1)
    data['bill_trend'] = data[[f'bill_time_{i}' for i in range(6)]].mean(axis=1) - data['avg_bill_amount']
    data.drop([f'bill_time_{i}' for i in range(6)], axis=1, inplace=True)
    
    # 2. Payment Amount Features (from PAY_AMT1-6)
    pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    data['total_paid'] = data[pay_cols].sum(axis=1)
    data['avg_payment'] = data[pay_cols].mean(axis=1)
    data['max_payment'] = data[pay_cols].max(axis=1)
    data['payment_volatility'] = data[pay_cols].std(axis=1)
    
    # Payment ratio (critical metric)
    data['payment_ratio'] = data['total_paid'] / (data[bill_cols].sum(axis=1) + 1)
    data['payment_to_income'] = data['total_paid'] / (data['LIMIT_BAL'] + 1)
    
    # 3. Payment Status Features (from PAY_0-6)
    pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    data['max_delay'] = data[pay_status_cols].max(axis=1)
    data['avg_delay'] = data[pay_status_cols].mean(axis=1)
    data['num_late_payments'] = (data[pay_status_cols] > 0).sum(axis=1)
    data['num_on_time_payments'] = (data[pay_status_cols] == 0).sum(axis=1)
    data['delay_volatility'] = data[pay_status_cols].std(axis=1)
    data['worst_payment_status'] = data[pay_status_cols].max(axis=1)
    
    # 4. Ratio and Interaction Features
    data['debt_to_income'] = data['BILL_AMT1'] / (data['LIMIT_BAL'] + 1)
    data['credit_utilization'] = data['avg_bill_amount'] / (data['LIMIT_BAL'] + 1)
    data['bill_to_income'] = data['max_bill'] / (data['LIMIT_BAL'] + 1)
    
    # Fill NaN values (from std calculations)
    data = data.fillna(0)
    
    # Select only the features used in training (in correct order)
    return data[feature_names]


def predict_default(data: Dict) -> float:
    """
    Predict default probability using calibrated model.
    
    Args:
        data: Dictionary with raw UCI input fields
        
    Returns:
        Calibrated probability of default (0-1)
    """
    # Engineer features
    X = engineer_features(data)
    
    # Get calibrated probability
    prob = calibrated_model.predict_proba(X)[0][1]
    
    return float(prob)
