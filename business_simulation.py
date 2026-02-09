"""
Business Simulation: Calculate actual profit/loss for different thresholds

This answers the only question that matters: "Does this model make money?"
"""
import pandas as pd
import numpy as np
import joblib
from src.model import engineer_features_batch


def load_test_data():
    """Load test set with true outcomes"""
    # Load full dataset
    df = pd.read_csv("data/uci/UCI_Credit_Card.csv")
    df = df.rename(columns={'default.payment.next.month': 'default'})
    
    # Use same split as training (20% test = last 6000 samples)
    # Note: In production, you'd save test indices during training
    test_size = int(len(df) * 0.2)
    test_df = df.tail(test_size).copy()
    
    return test_df


def simulate_business_outcome(threshold: float, test_data: pd.DataFrame, 
                              model, interest_rate: float = 0.15) -> dict:
    """
    Simulate business outcomes for a given decision threshold.
    
    Args:
        threshold: Probability threshold for approval (prob < threshold → approve)
        test_data: Test dataset with true outcomes
        model: Trained model
        interest_rate: Annual interest rate (default 15%)
        
    Returns:
        Dict with business metrics
    """
    # Engineer features
    features = engineer_features_batch(test_data)
    
    # Get predictions
    probabilities = model.predict_proba(features)[:, 1]
    
    # Decision: approve if prob < threshold
    approved = probabilities < threshold
    
    # Get loan amounts and true outcomes
    loan_amounts = test_data['LIMIT_BAL'].values
    true_defaults = test_data['default'].values
    
    # Calculate profit per customer
    profits = np.zeros(len(test_data))
    
    for i in range(len(test_data)):
        if approved[i]:
            if true_defaults[i] == 1:
                # Approved but defaulted → lose full loan
                profits[i] = -1.0 * loan_amounts[i]
            else:
                # Approved and paid back → earn interest
                profits[i] = interest_rate * loan_amounts[i]
        else:
            # Rejected → no profit, no loss
            profits[i] = 0
    
    # Calculate metrics for approved loans only
    approved_indices = np.where(approved)[0]
    if len(approved_indices) > 0:
        defaults_among_approved = true_defaults[approved_indices].sum()
        default_rate_approved = defaults_among_approved / len(approved_indices)
        total_approved_amount = loan_amounts[approved_indices].sum()
    else:
        defaults_among_approved = 0
        default_rate_approved = 0
        total_approved_amount = 0
    
    # Calculate financial outcomes
    interest_earned = profits[profits > 0].sum()
    losses_incurred = profits[profits < 0].sum()
    net_profit = profits.sum()
    
    return {
        'threshold': threshold,
        'total_customers': len(test_data),
        'approved_count': approved.sum(),
        'approval_rate': approved.sum() / len(test_data),
        'rejected_count': (~approved).sum(),
        'defaults_among_approved': int(defaults_among_approved),
        'default_rate_approved': default_rate_approved,
        'total_approved_amount': total_approved_amount,
        'interest_earned': interest_earned,
        'losses_incurred': losses_incurred,
        'net_profit': net_profit,
        'roi': (net_profit / total_approved_amount * 100) if total_approved_amount > 0 else 0
    }


def run_simulation(thresholds=[0.20, 0.30, 0.50]):
    """
    Run business simulation across multiple thresholds.
    
    Args:
        thresholds: List of probability thresholds to test
    """
    print("\n" + "="*70)
    print("BUSINESS SIMULATION: Profit/Loss Analysis")
    print("="*70)
    print("\nAssumptions:")
    print("  • Loan amount = customer's credit limit (LIMIT_BAL)")
    print("  • Interest rate = 15% annually")
    print("  • If customer defaults → lose full loan amount")
    print("  • If customer repays → earn 15% interest")
    print("  • If rejected → $0 profit/loss")
    print("\n" + "="*70)
    
    # Load model and test data
    print("\nLoading model and test data...")
    model = joblib.load("src/uci_model_calibrated.pkl")
    test_data = load_test_data()
    print(f"✓ Loaded {len(test_data)} test samples")
    
    results = []
    
    for threshold in thresholds:
        print(f"\nSimulating threshold {threshold:.2f}...")
        result = simulate_business_outcome(threshold, test_data, model)
        results.append(result)
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    for result in results:
        print(f"\n{'─'*70}")
        print(f"Threshold: {result['threshold']:.2f}")
        print(f"{'─'*70}")
        print(f"  Approved Customers:        {result['approved_count']:,} / {result['total_customers']:,} ({result['approval_rate']:.1%})")
        print(f"  Rejected Customers:        {result['rejected_count']:,} ({(1-result['approval_rate']):.1%})")
        print(f"  Total Loan Amount:         ${result['total_approved_amount']:,.0f}")
        print(f"\n  Defaults (among approved): {result['defaults_among_approved']:,} ({result['default_rate_approved']:.1%})")
        print(f"\n  Interest Earned:           ${result['interest_earned']:,.0f}")
        print(f"  Losses from Defaults:      ${result['losses_incurred']:,.0f}")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Color code profit/loss
        if result['net_profit'] >= 0:
            profit_label = f"  NET PROFIT:                ${result['net_profit']:,.0f} ✓"
        else:
            profit_label = f"  NET LOSS:                  ${result['net_profit']:,.0f} ✗"
        print(profit_label)
        print(f"  ROI:                       {result['roi']:.2f}%")
    
    # Find optimal threshold
    print("\n" + "="*70)
    print("BUSINESS INSIGHTS")
    print("="*70)
    
    best_result = max(results, key=lambda x: x['net_profit'])
    
    print(f"\n✓ Most Profitable Threshold: {best_result['threshold']:.2f}")
    print(f"  Net Profit: ${best_result['net_profit']:,.0f}")
    print(f"  Approval Rate: {best_result['approval_rate']:.1%}")
    print(f"  Default Rate: {best_result['default_rate_approved']:.1%}")
    
    # Compare with current operational threshold (0.20)
    current_result = next(r for r in results if r['threshold'] == 0.20)
    optimal_result = best_result
    
    if optimal_result['threshold'] != 0.20:
        profit_gain = optimal_result['net_profit'] - current_result['net_profit']
        print(f"\n⚠️  Current threshold (0.20) leaves ${profit_gain:,.0f} on the table!")
        print(f"   Recommendation: Increase threshold to {optimal_result['threshold']:.2f}")
        print(f"   Trade-off: Approve {current_result['approval_rate']:.1%} → {optimal_result['approval_rate']:.1%} customers")
    else:
        print(f"\n✓ Current threshold (0.20) is already optimal!")
    
    print("\n" + "="*70)
    print("Key Takeaway:")
    print("  Aggressive lending (low threshold) → High volume, high risk, potential losses")
    print("  Conservative lending (high threshold) → Low volume, low risk, higher profit")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    # Run simulation with standard thresholds
    results = run_simulation(thresholds=[0.20, 0.30, 0.50])
