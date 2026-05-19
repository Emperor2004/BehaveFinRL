import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def prospect_theory_utility(pnl, alpha=config.PROSPECT_ALPHA, beta=config.PROSPECT_BETA, lam=config.PROSPECT_LAMBDA):
    """
    Calculates the Kahneman-Tversky Prospect Theory utility for a given P&L.
    Supports both scalar and numpy array inputs.
    
    Parameters:
    - pnl: float or np.ndarray, the raw profit/loss return signal.
    - alpha: float, value function exponent for gains (concave part).
    - beta: float, value function exponent for losses (convex part).
    - lam: float, loss aversion coefficient (scaling factor for losses).
    
    Returns:
    - utility: same type as pnl, the prospect-theoretic utility value.
    """
    # Handle numpy arrays
    if isinstance(pnl, np.ndarray):
        utility = np.zeros_like(pnl, dtype=float)
        
        # Gains: concave shape (x >= 0)
        gain_mask = (pnl >= 0)
        utility[gain_mask] = np.power(pnl[gain_mask], alpha)
        
        # Losses: convex shape with loss aversion lambda (x < 0)
        loss_mask = (pnl < 0)
        utility[loss_mask] = -lam * np.power(-pnl[loss_mask], beta)
        
        return utility
        
    # Handle scalar inputs
    else:
        if pnl >= 0:
            return float(np.power(pnl, alpha))
        else:
            return float(-lam * np.power(-pnl, beta))

if __name__ == "__main__":
    # Small test
    test_gains = [0.0, 0.01, 0.05, 0.10]
    test_losses = [-0.01, -0.05, -0.10]
    
    print("Testing Prospect Theory Reward Shaping:")
    print("Gains:")
    for g in test_gains:
        print(f"  Raw: {g:+.2f} -> Utility: {prospect_theory_utility(g):+.4f}")
    print("Losses (Default Lambda = 2.25):")
    for l in test_losses:
        print(f"  Raw: {l:+.2f} -> Utility: {prospect_theory_utility(l):+.4f}")
        
    # Test regime adaptive lambda
    print("Losses (Bear Lambda = 2.75):")
    for l in test_losses:
        print(f"  Raw: {l:+.2f} -> Utility: {prospect_theory_utility(l, lam=config.LAMBDA_BEAR):+.4f}")
