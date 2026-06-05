import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.reward import (
    prospect_theory_utility,
    probability_weight,
    prelec_weight,
    var_penalty_weight,
    weighted_prospect_utility,
)

def test_reward_neutral():
    # P&L = 0 must return 0
    assert prospect_theory_utility(0.0) == 0.0
    
def test_reward_gains():
    # Gain of 0.05
    r = prospect_theory_utility(0.05)
    assert r > 0.0
    # Concave shape test: v(2x) < 2v(x)
    r1 = prospect_theory_utility(0.02)
    r2 = prospect_theory_utility(0.04)
    assert r2 < 2.0 * r1

def test_reward_losses():
    # Loss of 0.05
    r = prospect_theory_utility(-0.05)
    assert r < 0.0
    # Loss aversion penalty: v(-x) should be much worse than -v(x)
    r_gain = prospect_theory_utility(0.05)
    r_loss = prospect_theory_utility(-0.05)
    assert abs(r_loss) > r_gain * 2.0  # since lambda = 2.25

def test_regime_adaptive_losses():
    # Loss of 0.05 under Bull regime (lambda = 2.00) vs Bear regime (lambda = 2.75)
    r_bull = prospect_theory_utility(-0.05, lam=2.00)
    r_bear = prospect_theory_utility(-0.05, lam=2.75)
    assert r_bear < r_bull  # Bear loss is penalized more severely (more negative)


# ---------------------------------------------------------------------------
# probability_weight tests (Task 2.1 — Requirements 1.1, 1.2, 1.3, 1.5)
# ---------------------------------------------------------------------------

def test_probability_weight_boundary_zero():
    """Requirement 1.2: p == 0.0 must return exactly 0.0."""
    assert probability_weight(0.0) == 0.0


def test_probability_weight_boundary_one():
    """Requirement 1.3: p == 1.0 must return exactly 1.0."""
    assert probability_weight(1.0) == 1.0


def test_probability_weight_formula():
    """Requirement 1.1: formula π(p) = p^γ / (p^γ + (1−p)^γ)^(1/γ)."""
    p, g = 0.3, 0.65
    expected = p ** g / (p ** g + (1.0 - p) ** g) ** (1.0 / g)
    assert abs(probability_weight(p, g) - expected) < 1e-10


def test_probability_weight_default_gamma():
    """Default gamma should equal config.PROB_WEIGHT_GAMMA (0.65)."""
    p = 0.5
    result_default = probability_weight(p)
    result_explicit = probability_weight(p, config.PROB_WEIGHT_GAMMA)
    assert result_default == result_explicit


def test_probability_weight_invalid_below_zero():
    """Requirement 1.5: p < 0 must raise ValueError."""
    with pytest.raises(ValueError, match="p must be in"):
        probability_weight(-0.1)


def test_probability_weight_invalid_above_one():
    """Requirement 1.5: p > 1 must raise ValueError."""
    with pytest.raises(ValueError, match="p must be in"):
        probability_weight(1.1)


# ---------------------------------------------------------------------------
# prelec_weight tests (Task 2.2 — Requirements 1.4, 1.5)
# ---------------------------------------------------------------------------

def test_prelec_weight_formula():
    """Requirement 1.4: formula π(p) = exp(−(−ln p)^γ)."""
    p, g = 0.05, 0.65
    expected = float(np.exp(-((-np.log(p)) ** g)))
    assert abs(prelec_weight(p, g) - expected) < 1e-10


def test_prelec_weight_at_one():
    """prelec_weight(1.0) should equal 1.0 (exp(0) = 1)."""
    assert abs(prelec_weight(1.0) - 1.0) < 1e-10


def test_prelec_weight_default_gamma():
    """Default gamma should equal config.PROB_WEIGHT_GAMMA (0.65)."""
    p = 0.3
    assert prelec_weight(p) == prelec_weight(p, config.PROB_WEIGHT_GAMMA)


def test_prelec_weight_invalid_zero():
    """Requirement 1.5: p == 0 must raise ValueError (domain is (0, 1])."""
    with pytest.raises(ValueError, match="p must be in"):
        prelec_weight(0.0)


def test_prelec_weight_invalid_above_one():
    """Requirement 1.5: p > 1 must raise ValueError."""
    with pytest.raises(ValueError, match="p must be in"):
        prelec_weight(1.1)


# ---------------------------------------------------------------------------
# var_penalty_weight tests (Task 2.3 — Requirement 2.4)
# ---------------------------------------------------------------------------

def test_var_penalty_weight_equals_prelec_of_tail():
    """Requirement 2.4: var_penalty_weight(gamma) == prelec_weight(0.05, gamma)."""
    g = 0.65
    expected = prelec_weight(0.05, g)
    assert abs(var_penalty_weight(g) - expected) < 1e-10


def test_var_penalty_weight_default_gamma():
    """Default gamma should use config.PROB_WEIGHT_GAMMA."""
    assert abs(var_penalty_weight() - prelec_weight(0.05, config.PROB_WEIGHT_GAMMA)) < 1e-10


def test_var_penalty_weight_uses_var_alpha():
    """var_penalty_weight should use 1.0 - config.VAR_ALPHA as the tail probability."""
    tail = 1.0 - config.VAR_ALPHA  # 0.05
    g = 0.65
    assert abs(var_penalty_weight(g) - prelec_weight(tail, g)) < 1e-10


# ---------------------------------------------------------------------------
# weighted_prospect_utility tests (Task 2.4 — Requirements 2.1, 2.2, 2.3, 2.5, 2.6)
# ---------------------------------------------------------------------------

def test_weighted_utility_gain():
    """Requirement 2.2: gain = probability_weight(win_rate, gamma) * pnl^alpha."""
    pnl, wr = 0.05, 0.6
    alpha = config.PROSPECT_ALPHA
    gamma = config.PROB_WEIGHT_GAMMA
    pi = probability_weight(wr, gamma)
    expected = pi * (pnl ** alpha)
    result = weighted_prospect_utility(pnl, wr)
    assert abs(result - expected) < 1e-10


def test_weighted_utility_loss():
    """Requirement 2.3: loss = -lam * probability_weight(1 - win_rate, gamma) * (-pnl)^beta."""
    pnl, wr = -0.05, 0.6
    beta = config.PROSPECT_BETA
    lam = config.PROSPECT_LAMBDA
    gamma = config.PROB_WEIGHT_GAMMA
    pi = probability_weight(1.0 - wr, gamma)
    expected = -lam * pi * ((-pnl) ** beta)
    result = weighted_prospect_utility(pnl, wr)
    assert abs(result - expected) < 1e-10


def test_weighted_utility_win_rate_clamping():
    """Requirement 2.5: out-of-range win_rate must not raise an exception."""
    # Should not raise for win_rate below 0
    weighted_prospect_utility(0.01, win_rate=-0.5)
    # Should not raise for win_rate above 1
    weighted_prospect_utility(0.01, win_rate=1.5)
