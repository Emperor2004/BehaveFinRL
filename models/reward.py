import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def probability_weight(p: float, gamma: float = config.PROB_WEIGHT_GAMMA) -> float:
    """
    Tversky-Kahneman (1992) probability weighting function.

    π(p) = p^γ / (p^γ + (1−p)^γ)^(1/γ)

    Boundary cases:
    - p == 0.0 → 0.0
    - p == 1.0 → 1.0

    Parameters:
    - p: float, probability in [0, 1].
    - gamma: float, curvature parameter γ ∈ (0, 1]. Default: config.PROB_WEIGHT_GAMMA.

    Returns:
    - float, the probability weight π(p).

    Raises:
    - ValueError: if p < 0 or p > 1.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    pg = p ** gamma
    return float(pg / (pg + (1.0 - p) ** gamma) ** (1.0 / gamma))


def prelec_weight(p: float, gamma: float = config.PROB_WEIGHT_GAMMA) -> float:
    """
    Prelec (1998) single-parameter probability weighting function.

    π(p) = exp(−(−ln p)^γ)

    Parameters:
    - p: float, probability in (0, 1].
    - gamma: float, curvature parameter γ ∈ (0, 1]. Default: config.PROB_WEIGHT_GAMMA.

    Returns:
    - float, the Prelec probability weight π(p).

    Raises:
    - ValueError: if p <= 0 or p > 1.
    """
    if p <= 0.0 or p > 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")
    return float(np.exp(-(-np.log(p)) ** gamma))


def var_penalty_weight(gamma: float = config.PROB_WEIGHT_GAMMA) -> float:
    """
    Returns the Prelec probability weight of the VaR tail probability (0.05),
    used to amplify rare-loss penalties in the reward function.

    Equivalent to prelec_weight(1.0 − config.VAR_ALPHA, gamma) = prelec_weight(0.05, gamma).

    Parameters:
    - gamma: float, curvature parameter γ ∈ (0, 1]. Default: config.PROB_WEIGHT_GAMMA.

    Returns:
    - float, the Prelec weight of the 0.05 tail probability.
    """
    var_tail_prob = 1.0 - config.VAR_ALPHA  # = 0.05
    return prelec_weight(var_tail_prob, gamma)


def weighted_prospect_utility(
    pnl: float,
    win_rate: float,
    var_tail_prob: float = 0.05,
    alpha: float = config.PROSPECT_ALPHA,
    beta: float = config.PROSPECT_BETA,
    lam: float = config.PROSPECT_LAMBDA,
    gamma: float = config.PROB_WEIGHT_GAMMA,
) -> float:
    """
    Prospect Theory utility with probability-weighted gain/loss components.

    Gain branch (pnl >= 0):
        π(win_rate, γ) * pnl^α

    Loss branch (pnl < 0):
        −λ * π(1 − win_rate, γ) * (−pnl)^β

    where π is the Tversky-Kahneman probability weighting function.

    Parameters:
    - pnl: float, the profit/loss signal (relative to reference point).
    - win_rate: float, rolling win-rate; clamped to [0.001, 0.999] internally.
    - var_tail_prob: float, VaR tail probability (accepted but not used in the
      value function itself; used externally for VaR penalty scaling). Default: 0.05.
    - alpha: float, gain exponent. Default: config.PROSPECT_ALPHA.
    - beta: float, loss exponent. Default: config.PROSPECT_BETA.
    - lam: float, loss aversion coefficient. Default: config.PROSPECT_LAMBDA.
    - gamma: float, probability weighting curvature. Default: config.PROB_WEIGHT_GAMMA.

    Returns:
    - float, the probability-weighted PT utility.
    """
    # Clamp win_rate to avoid log-domain errors in probability_weight
    win_rate = float(np.clip(win_rate, 0.001, 0.999))

    if pnl >= 0.0:
        pi_gain = probability_weight(win_rate, gamma)
        return float(pi_gain * (pnl ** alpha))
    else:
        pi_loss = probability_weight(1.0 - win_rate, gamma)
        return float(-lam * pi_loss * ((-pnl) ** beta))


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
