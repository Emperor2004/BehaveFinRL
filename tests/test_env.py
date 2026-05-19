import sys
import numpy as np
import pandas as pd
from pathlib import Path
import gymnasium as gym

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from env.trading_env import TradingEnv

def test_env_initialization():
    # Create a mock dataframe containing required preprocessing columns
    mock_df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "adj_close": [100.0, 105.0, 95.0, 100.0],
        "vix": [15.0, 15.2, 16.0, 15.5],
        "rolling_volatility": [0.01, 0.01, 0.01, 0.01],
        "bid_ask_spread": [0.0005, 0.0005, 0.0005, 0.0005],
        "order_book_imbalance": [0.1, 0.2, 0.3, 0.4]
    })
    
    env = TradingEnv(mock_df, initial_balance=10000.0)
    obs, info = env.reset()
    
    # Dimension verification:
    # Feature columns exclude: date, open, high, low, close, adj_close, volume, daily_return
    # Remaining: vix, rolling_volatility, bid_ask_spread, order_book_imbalance = 4
    # State features: portfolio_weight, last_return = 2
    # Total = 6
    assert len(obs) == 6
    assert env.balance == 10000.0
    assert env.shares == 0.0
    assert env.holding_state == 0.0

def test_env_continuous_weight_transitions():
    mock_df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "adj_close": [100.0, 110.0, 99.0, 100.0],
        "vix": [15.0, 15.2, 16.0, 15.5],
        "rolling_volatility": [0.015, 0.015, 0.015, 0.015],
        "bid_ask_spread": [0.0005, 0.0005, 0.0005, 0.0005],
        "order_book_imbalance": [0.1, 0.2, 0.3, 0.4]
    })
    
    env = TradingEnv(mock_df, initial_balance=10000.0)
    obs, info = env.reset()
    
    # 1. Allocate 100% Long: action = [1.0]
    obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    
    # Friction verification:
    # weight change delta = 1.0 (from 0.0 to 1.0)
    # fee = 1.0 * TRANSACTION_FEE_PCT (0.0015)
    # slippage = 1.0 * (0.5 * bid_ask_spread + SLIPPAGE_COEF * rolling_volatility)
    #          = 1.0 * (0.00025 + 0.05 * 0.015) = 0.001
    # market impact = MARKET_IMPACT_COEF * delta^2 = 0.02 * (1.0)^2 = 0.02
    # total drag = 0.0015 + 0.001 + 0.02 = 0.0225
    # asset return = 110/100 - 1.0 = 0.10
    # portfolio net return = 1.0 * 0.10 - 0.0225 = 0.0775
    # net worth = 10000 * 1.0775 = 10775.0
    assert abs(env.net_worth - 10775.0) < 1.0
    assert env.holding_state > 0.95
    
    # 2. Rebalance to Neutral: action = [0.0]
    obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    # We transitioned to cash, so shares should be 0.0
    assert abs(env.shares) < 1e-4
