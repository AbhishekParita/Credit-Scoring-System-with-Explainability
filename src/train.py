
import pandas as pd
import numpy as np
import joblib
import random
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report, confusion_matrix
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from src.calibration import CalibratedClassifier

# ============================================================================
# PHASE 0: Setup & Configuration
# ============================================================================

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Feature definitions (will be populated after feature engineering)
FEATURES = []
TARGET = "default"

# Cost-sensitive configuration
COST_FN = 5  # Cost of missing a defaulter (False Negative)
COST_FP = 1  # Cost of rejecting a good customer (False Positive)

# ============================================================================
# PHASE 1: Data Loading & Understanding
# ============================================================================

print("="*60)
print("PHASE 1: DATA LOADING & EXPLORATION")
print("="*60)

# Load data with proper path separator
df = pd.read_csv("data/uci/UCI_Credit_Card.csv")

print(f"\nDataset Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ============================================================================
# PHASE 2: Data Preparation & Feature Engineering
# ============================================================================

print("\n" + "="*60)
print("PHASE 2: DATA PREPARATION & FEATURE ENGINEERING")
print("="*60)

# Rename target column only (keep original feature names for now)
if "default.payment.next.month" in df.columns:
    df = df.rename(columns={"default.payment.next.month": "default"})

print("\n✓ Target column renamed")

# ============================================================================
# ADVANCED FEATURE ENGINEERING (CRITICAL FOR PERFORMANCE)
# ============================================================================

print("\n" + "-"*60)
print("FEATURE ENGINEERING (using original UCI column names)")
print("-"*60)

# 1. Bill Amount Features (from BILL_AMT1-6)
bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
df['avg_bill_amount'] = df[bill_cols].mean(axis=1)
df['max_bill'] = df[bill_cols].max(axis=1)
df['min_bill'] = df[bill_cols].min(axis=1)
df['bill_volatility'] = df[bill_cols].std(axis=1)

# Bill trend (slope over 6 months)
for i, col in enumerate(bill_cols):
    df[f'bill_time_{i}'] = df[col] * (i + 1)  # Weighted by time
df['bill_trend'] = df[[f'bill_time_{i}' for i in range(6)]].mean(axis=1) - df['avg_bill_amount']
df.drop([f'bill_time_{i}' for i in range(6)], axis=1, inplace=True)

print("✓ Bill amount features created")

# 2. Payment Amount Features (from PAY_AMT1-6)
pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
df['total_paid'] = df[pay_cols].sum(axis=1)
df['avg_payment'] = df[pay_cols].mean(axis=1)
df['max_payment'] = df[pay_cols].max(axis=1)
df['payment_volatility'] = df[pay_cols].std(axis=1)

# Payment ratio (critical metric)
df['payment_ratio'] = df['total_paid'] / (df[bill_cols].sum(axis=1) + 1)  # Avoid division by zero
df['payment_to_income'] = df['total_paid'] / (df['LIMIT_BAL'] + 1)

print("✓ Payment amount features created")

# 3. Payment Status Features (from PAY_0-6)
pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
df['max_delay'] = df[pay_status_cols].max(axis=1)
df['avg_delay'] = df[pay_status_cols].mean(axis=1)
df['num_late_payments'] = (df[pay_status_cols] > 0).sum(axis=1)
df['num_on_time_payments'] = (df[pay_status_cols] == 0).sum(axis=1)
df['delay_volatility'] = df[pay_status_cols].std(axis=1)

# Worst payment status
df['worst_payment_status'] = df[pay_status_cols].max(axis=1)

print("✓ Payment status features created")

# 4. Ratio and Interaction Features
df['debt_to_income'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
df['credit_utilization'] = df['avg_bill_amount'] / (df['LIMIT_BAL'] + 1)
df['bill_to_income'] = df['max_bill'] / (df['LIMIT_BAL'] + 1)

print("✓ Ratio features created")

# Define final feature list (using original + engineered features)
FEATURES = [
    # Original UCI features (keeping original names)
    'LIMIT_BAL', 'AGE', 'EDUCATION',
    
    # Bill features
    'avg_bill_amount', 'max_bill', 'min_bill', 'bill_volatility', 'bill_trend',
    
    # Payment features
    'total_paid', 'avg_payment', 'max_payment', 'payment_volatility',
    'payment_ratio', 'payment_to_income',
    
    # Payment status features
    'max_delay', 'avg_delay', 'num_late_payments', 'num_on_time_payments',
    'delay_volatility', 'worst_payment_status', 'PAY_0',
    
    # Ratio features
    'debt_to_income', 'credit_utilization', 'bill_to_income'
]

print(f"\n✓ Total engineered features: {len(FEATURES)}")

# Data validation
print(f"\nMissing values in engineered features:")
missing_counts = df[FEATURES + [TARGET]].isnull().sum()
if missing_counts.sum() > 0:
    print(missing_counts[missing_counts > 0])
    # Fill NaN with 0 (from std calculations on single values)
    df[FEATURES] = df[FEATURES].fillna(0)
    print("✓ NaN values filled with 0")
else:
    print("None")

print(f"\nDuplicates: {df.duplicated().sum()}")

# Check class balance (critical for credit scoring)
print(f"\nClass Distribution:")
print(df[TARGET].value_counts())
print(f"\nClass Balance (%):")
print(df[TARGET].value_counts(normalize=True) * 100)

# Basic statistics
print(f"\nTop 10 Feature Statistics:")
print(df[FEATURES[:10]].describe())

# Extract features and target
X = df[FEATURES]
y = df[TARGET]

print(f"\n✓ Final dataset: {X.shape[0]} samples, {X.shape[1]} features")

# ============================================================================
# PHASE 3: Data Splitting Strategy (Stratified)
# ============================================================================

print("\n" + "="*60)
print("PHASE 3: DATA SPLITTING (STRATIFIED)")
print("="*60)

# Three-way split: 60% train, 20% validation, 20% test
# Use stratification to preserve class balance
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=RANDOM_SEED, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Verify stratification preserved class balance
print(f"\nClass distribution verification:")
print(f"Train: {y_train.value_counts(normalize=True).values}")
print(f"Val:   {y_val.value_counts(normalize=True).values}")
print(f"Test:  {y_test.value_counts(normalize=True).values}")

# ============================================================================
# PHASE 4: Baseline Models
# ============================================================================

print("\n" + "="*60)
print("PHASE 4: BASELINE MODELS")
print("="*60)

baselines = {
    'majority_class': DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED),
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
}

baseline_results = {}

for name, model in baselines.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    }
    
    baseline_results[name] = results
    print(f"\n{name}:")
    for metric, value in results.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")

# ============================================================================
# PHASE 5: Model Training (XGBoost with Best Practices)
# ============================================================================

print("\n" + "="*60)
print("PHASE 5: XGBOOST MODEL TRAINING")
print("="*60)

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nClass imbalance ratio: {scale_pos_weight:.2f}")
print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

# Initialize model with proper configuration
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    objective='binary:logistic',  # Explicitly set binary classification
    scale_pos_weight=scale_pos_weight,  # Handle imbalance
    eval_metric='logloss',
    early_stopping_rounds=20,  # Early stopping in model initialization
    random_state=RANDOM_SEED,
    use_label_encoder=False
)

# Train with early stopping on validation set
print("\nTraining XGBoost with early stopping...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

print(f"✓ Training complete (best iteration: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'})")

# ============================================================================
# PHASE 5.5: Probability Calibration
# ============================================================================

print("\n" + "="*60)
print("PHASE 5.5: PROBABILITY CALIBRATION")
print("="*60)

print("\nCalibrating probabilities using isotonic regression...")
# Use custom calibrator for sklearn 1.6+ compatibility
calibrated_model = CalibratedClassifier(model)
calibrated_model.fit(X_val, y_val)
print("✓ Calibration complete")

# ============================================================================
# PHASE 6: Model Evaluation
# ============================================================================

print("\n" + "="*60)
print("PHASE 6: MODEL EVALUATION")
print("="*60)

# Predictions (using calibrated model)
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

y_train_proba = model.predict_proba(X_train)[:, 1]
y_val_proba = model.predict_proba(X_val)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]

# Calibrated predictions
y_train_proba_cal = calibrated_model.predict_proba(X_train)[:, 1]
y_val_proba_cal = calibrated_model.predict_proba(X_val)[:, 1]
y_test_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]

# Comprehensive metrics
print("\n" + "-"*60)
print("TRAIN SET METRICS")
print("-"*60)
print(f"Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Precision: {precision_score(y_train, y_train_pred):.4f}")
print(f"Recall:    {recall_score(y_train, y_train_pred):.4f}")
print(f"F1 Score:  {f1_score(y_train, y_train_pred):.4f}")
print(f"AUC:       {roc_auc_score(y_train, y_train_proba):.4f}")

print("\n" + "-"*60)
print("VALIDATION SET METRICS")
print("-"*60)
print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall:    {recall_score(y_val, y_val_pred):.4f}")
print(f"F1 Score:  {f1_score(y_val, y_val_pred):.4f}")
print(f"AUC:       {roc_auc_score(y_val, y_val_proba):.4f}")

print("\n" + "-"*60)
print("TEST SET METRICS (FINAL)")
print("-"*60)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1 Score:  {test_f1:.4f}")
print(f"AUC:       {test_auc:.4f}")

# Detailed classification report
print("\n" + "-"*60)
print("CLASSIFICATION REPORT")
print("-"*60)
print(classification_report(y_test, y_test_pred, target_names=['No Default', 'Default']))

# Confusion matrix
print("\n" + "-"*60)
print("CONFUSION MATRIX")
print("-"*60)
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"\nTrue Negatives:  {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives:  {cm[1,1]}")

# Feature importance
print("\n" + "-"*60)
print("FEATURE IMPORTANCE")
print("-"*60)
feature_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nTop 15 Most Important Features (out of {len(FEATURES)}):")
print(feature_importance.head(15).to_string(index=False))

# ============================================================================
# PHASE 6.5: Threshold Optimization & Cost-Sensitive Analysis
# ============================================================================

print("\n" + "="*60)
print("PHASE 6.5: THRESHOLD OPTIMIZATION & COST-SENSITIVE ANALYSIS")
print("="*60)

# Define cost function
def calculate_cost(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    return total_cost, fn, fp

print(f"\nCost Configuration:")
print(f"  Cost of False Negative (missed defaulter): {COST_FN}x")
print(f"  Cost of False Positive (rejected good customer): {COST_FP}x")
print(f"  Ratio: Missing a defaulter is {COST_FN}x more expensive")

# Test different thresholds on validation set
thresholds = np.arange(0.2, 0.6, 0.05)
threshold_results = []

print("\n" + "-"*60)
print("Testing Thresholds on Validation Set")
print("-"*60)
print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Cost':<12}")
print("-"*60)

for threshold in thresholds:
    y_val_pred_thresh = (y_val_proba_cal >= threshold).astype(int)
    
    precision = precision_score(y_val, y_val_pred_thresh, zero_division=0)
    recall = recall_score(y_val, y_val_pred_thresh)
    f1 = f1_score(y_val, y_val_pred_thresh)
    cost, fn, fp = calculate_cost(y_val, y_val_pred_thresh)
    
    threshold_results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cost': cost,
        'fn': fn,
        'fp': fp
    })
    
    print(f"{threshold:<12.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {cost:<12.0f}")

# Find optimal threshold (minimum cost)
threshold_df = pd.DataFrame(threshold_results)
optimal_idx = threshold_df['cost'].idxmin()
optimal_threshold = threshold_df.loc[optimal_idx, 'threshold']
optimal_cost = threshold_df.loc[optimal_idx, 'cost']

print("\n" + "-"*60)
print(f"OPTIMAL THRESHOLD: {optimal_threshold:.2f}")
print(f"MINIMUM COST: {optimal_cost:.0f}")
print("-"*60)

# Apply optimal threshold to test set
y_test_pred_optimal = (y_test_proba_cal >= optimal_threshold).astype(int)

print("\n" + "-"*60)
print("TEST SET METRICS with OPTIMAL THRESHOLD")
print("-"*60)

test_cost, test_fn, test_fp = calculate_cost(y_test, y_test_pred_optimal)

print(f"Threshold:  {optimal_threshold:.2f}")
print(f"Accuracy:   {accuracy_score(y_test, y_test_pred_optimal):.4f}")
print(f"Precision:  {precision_score(y_test, y_test_pred_optimal):.4f}")
print(f"Recall:     {recall_score(y_test, y_test_pred_optimal):.4f}")
print(f"F1 Score:   {f1_score(y_test, y_test_pred_optimal):.4f}")
print(f"AUC:        {test_auc:.4f} (unchanged - threshold-independent)")
print(f"\nCost Analysis:")
print(f"  False Negatives: {test_fn} × {COST_FN} = {test_fn * COST_FN}")
print(f"  False Positives: {test_fp} × {COST_FP} = {test_fp * COST_FP}")
print(f"  Total Cost:      {test_cost}")

# Comparison with default threshold (0.5)
y_test_pred_default = (y_test_proba_cal >= 0.5).astype(int)
default_cost, default_fn, default_fp = calculate_cost(y_test, y_test_pred_default)

print(f"\n" + "-"*60)
print("COST IMPROVEMENT vs DEFAULT THRESHOLD")
print("-"*60)
print(f"Default threshold (0.5) cost:   {default_cost}")
print(f"Optimized threshold ({optimal_threshold:.2f}) cost: {test_cost}")
print(f"Cost reduction:                  {default_cost - test_cost} ({((default_cost - test_cost) / default_cost * 100):.1f}%)")
print(f"\nFalse Negatives reduction:       {default_fn - test_fn} (from {default_fn} to {test_fn})")
print(f"False Positives change:          {test_fp - default_fp} (from {default_fp} to {test_fp})")

# Detailed comparison
print("\n" + "-"*60)
print("Confusion Matrix Comparison")
print("-"*60)
print("\nDefault Threshold (0.5):")
print(confusion_matrix(y_test, y_test_pred_default))
print(f"  TN: {confusion_matrix(y_test, y_test_pred_default)[0,0]}, FP: {default_fp}, FN: {default_fn}, TP: {confusion_matrix(y_test, y_test_pred_default)[1,1]}")

print(f"\nOptimal Threshold ({optimal_threshold:.2f}):")
cm_optimal = confusion_matrix(y_test, y_test_pred_optimal)
print(cm_optimal)
print(f"  TN: {cm_optimal[0,0]}, FP: {test_fp}, FN: {test_fn}, TP: {cm_optimal[1,1]}")

# ============================================================================
# PHASE 7: Model Persistence
# ============================================================================

print("\n" + "="*60)
print("PHASE 7: MODEL SAVING")
print("="*60)

# Save calibrated model (primary model for production)
model_path = "src/uci_model_calibrated.pkl"
joblib.dump(calibrated_model, model_path)
print(f"✓ Calibrated model saved to: {model_path}")

# Save base model (for SHAP explanations - works better with tree models)
base_model_path = "src/uci_model.pkl"
joblib.dump(model, base_model_path)
print(f"✓ Base model saved to: {base_model_path}")

# Save feature names
feature_path = "src/feature_names.pkl"
joblib.dump(FEATURES, feature_path)
print(f"✓ Feature names saved to: {feature_path}")

# Save optimal threshold
threshold_path = "src/optimal_threshold.pkl"
joblib.dump(optimal_threshold, threshold_path)
print(f"✓ Optimal threshold ({optimal_threshold:.2f}) saved to: {threshold_path}")

# Save comprehensive metrics
metrics = {
    'test_accuracy': test_accuracy,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1': test_f1,
    'test_auc': test_auc,
    'optimal_threshold': optimal_threshold,
    'optimal_cost': test_cost,
    'default_cost': default_cost,
    'cost_reduction_pct': (default_cost - test_cost) / default_cost * 100,
    'feature_importance': feature_importance.head(15).to_dict(),
    'num_features': len(FEATURES),
    'cost_fn': COST_FN,
    'cost_fp': COST_FP
}
metrics_path = "src/model_metrics.pkl"
joblib.dump(metrics, metrics_path)
print(f"✓ Metrics saved to: {metrics_path}")

# Save training statistics for drift monitoring
from src.monitor import save_training_stats, init_monitoring
max_delay_mean = df_engineered['max_delay'].mean()
save_training_stats(max_delay_mean)
init_monitoring()
print("✓ Monitoring initialized")

print("\n" + "="*60)
print("TRAINING PIPELINE COMPLETE ✓")
print("="*60)