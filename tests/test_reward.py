import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.reward import prospect_theory_utility

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
