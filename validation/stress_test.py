import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO

PROJECT_ROOT = Path("d:/Studies/DBM/BehaveFinRL")
sys.path.append(str(PROJECT_ROOT))

import config
from env.trading_env import TradingEnv
from regime.hmm import MarketRegimeDetector
from data.preprocess import calculate_technical_indicators

def generate_gbm_path(s0, mu, sigma, n_days, dt=1/252.0):
    """
    Generates a single price path using Geometric Brownian Motion.
    """
    np.random.seed(42)
    t = np.arange(1, n_days + 1)
    # Wiener process
    z = np.random.normal(0, 1, n_days)
    # Exponential path
    s = s0 * np.exp(np.cumsum((mu - 0.5 * (sigma**2)) * dt + sigma * np.sqrt(dt) * z))
    return np.insert(s, 0, s0)[:-1]

def generate_garch_path(s0, omega, alpha, beta, n_days, initial_vol=0.015):
    """
    Generates a single price path with GARCH(1,1) volatility clustering.
    r_t = sigma_t * epsilon_t
    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
    """
    np.random.seed(42)
    returns = np.zeros(n_days)
    vols = np.zeros(n_days)
    vols[0] = initial_vol
    
    # Simulate return series
    for t in range(1, n_days):
        epsilon = np.random.normal(0, 1)
        returns[t] = vols[t-1] * epsilon
        # Update variance
        var_t = omega + alpha * (returns[t]**2) + beta * (vols[t-1]**2)
        vols[t] = np.sqrt(max(1e-6, var_t))
        
    # Convert returns to prices
    prices = s0 * np.exp(np.cumsum(returns))
    return np.insert(prices, 0, s0)[:-1]

def construct_synthetic_df(prices, template_df):
    """
    Constructs a complete trading feature dataframe based on a price series.
    Uses historical macro variables (VIX, fed rate, yield spread) from the template.
    """
    n_days = len(prices)
    df = pd.DataFrame()
    df["date"] = pd.date_range(start="2026-01-01", periods=n_days)
    df["close"] = prices
    df["open"] = prices
    df["high"] = prices
    df["low"] = prices
    df["adj_close"] = prices
    df["volume"] = 1000000.0
    
    # Bootstrap macro values from template
    for col in ["vix", "yield_spread", "fed_rate"]:
        if col in template_df.columns:
            # Tile or pad the values
            vals = template_df[col].values
            if len(vals) >= n_days:
                df[col] = vals[:n_days]
            else:
                df[col] = np.resize(vals, n_days)
        else:
            # Set default values if not present
            df[col] = 20.0 if col == "vix" else (1.5 if col == "yield_spread" else 4.0)
            
    # Calculate microstructure variables
    np.random.seed(42)
    vix_factor = df["vix"] / 100.0
    noise_spread = np.random.exponential(scale=0.0001, size=len(df))
    df["bid_ask_spread"] = 0.0002 + 0.001 * vix_factor + noise_spread
    
    daily_returns = df["close"].pct_change().fillna(0.0).values
    obi = np.zeros(len(df))
    rho = 0.3
    beta = 10.0
    for t in range(1, len(df)):
        noise = np.random.normal(loc=0.0, scale=0.15)
        obi[t] = rho * obi[t-1] + (1.0 - rho) * np.tanh(beta * daily_returns[t]) + noise
    df["order_book_imbalance"] = np.clip(obi, -1.0, 1.0)
    
    # Calculate indicators
    df_features = calculate_technical_indicators(df)
    df_features = df_features.dropna().reset_index(drop=True)
    return df_features

def run_stress_test_evaluation(model, df, regime_detector):
    """
    Evaluates a model in the environment and returns key performance and safety metrics.
    """
    env = TradingEnv(df, regime_detector=regime_detector)
    obs, info = env.reset()
    
    done = False
    rewards = []
    dd_violations = 0
    var_violations = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        
        # Track constraint violations
        if info["drawdown"] > config.MAX_DRAWDOWN_LIMIT:
            dd_violations += 1
        if info["var_95"] > config.VAR_LIMIT:
            var_violations += 1
            
    # Calculate metrics
    net_worths = env.net_worth_history
    cum_return = (net_worths[-1] / net_worths[0]) - 1.0
    
    # Max drawdown
    peak = net_worths[0]
    max_dd = 0.0
    for nw in net_worths:
        if nw > peak:
            peak = nw
        dd = (peak - nw) / peak
        if dd > max_dd:
            max_dd = dd
            
    return {
        "final_return": cum_return,
        "max_drawdown": max_dd,
        "dd_violations": dd_violations,
        "var_violations": var_violations,
        "avg_reward": np.mean(rewards)
    }

def main():
    print("==================================================")
    print("      BehaveFinRL Stress-Testing Framework        ")
    print("==================================================")
    
    # Load required data and models
    test_path = PROJECT_ROOT / "data_cache" / f"{config.TICKER}_test.csv"
    hmm_path = PROJECT_ROOT / "saved_models" / "hmm_regime_detector.joblib"
    
    if not test_path.exists() or not hmm_path.exists():
        print("Error: Test data or HMM model not found! Run preprocess and training first.")
        return
        
    test_df = pd.read_csv(test_path)
    regime_detector = MarketRegimeDetector.load(hmm_path)
    
    # Load primary agent (seed 21 or first available seed)
    agent_path = PROJECT_ROOT / "saved_models" / f"ppo_agent_seed_21.zip"
    if not agent_path.exists():
        # Fall back to first zip file in saved_models
        zip_files = list((PROJECT_ROOT / "saved_models").glob("*.zip"))
        if len(zip_files) == 0:
            print("Error: No trained PPO models found in saved_models/.")
            return
        agent_path = zip_files[0]
        
    print(f"Loading trained PPO model from {agent_path.name}...")
    model = PPO.load(agent_path)
    
    n_days = 250  # Simulate ~1 year of trading days
    s0 = test_df["adj_close"].values[0]
    
    # 1. GBM Stress Test (Trending Bear/High Volatility)
    print("\n[STRESS TEST 1] Generating GBM path (Bear Market, mu=-10%, vol=25%)...")
    gbm_prices = generate_gbm_path(s0=s0, mu=-0.10, sigma=0.25, n_days=n_days)
    gbm_df = construct_synthetic_df(gbm_prices, test_df)
    gbm_metrics = run_stress_test_evaluation(model, gbm_df, regime_detector)
    
    # 2. GARCH Volatility Clustering Test
    # High persistence (alpha+beta close to 1), typical during stress
    print("\n[STRESS TEST 2] Generating GARCH(1,1) path (Vol Clustering, persistent stress)...")
    garch_prices = generate_garch_path(s0=s0, omega=0.00001, alpha=0.12, beta=0.85, n_days=n_days)
    garch_df = construct_synthetic_df(garch_prices, test_df)
    garch_metrics = run_stress_test_evaluation(model, garch_df, regime_detector)
    
    # 3. Historical Crisis Sub-period Evaluation (e.g. 2022 Bear Market)
    # The first 250 days of SP500_test represents the 2022 bear market trend
    print("\n[STRESS TEST 3] Evaluating on Historical 2022 Bear Market sub-period...")
    hist_bear_df = test_df.iloc[:250].reset_index(drop=True)
    hist_metrics = run_stress_test_evaluation(model, hist_bear_df, regime_detector)
    
    # Display Results Table
    print("\n==================================================")
    print("              Stress Test Report Summary          ")
    print("==================================================")
    print(f"{'Scenario':<25} | {'Cum. Return':<12} | {'Max DD':<8} | {'VaR Viol.':<9}")
    print("-" * 65)
    print(f"{'GBM (Bear, mu=-10%)':<25} | {gbm_metrics['final_return']*100:>10.2f}% | {gbm_metrics['max_drawdown']*100:>6.2f}% | {gbm_metrics['var_violations']:>9}")
    print(f"{'GARCH Vol Clustering':<25} | {garch_metrics['final_return']*100:>10.2f}% | {garch_metrics['max_drawdown']*100:>6.2f}% | {garch_metrics['var_violations']:>9}")
    print(f"{'2022 Bear Sub-period':<25} | {hist_metrics['final_return']*100:>10.2f}% | {hist_metrics['max_drawdown']*100:>6.2f}% | {hist_metrics['var_violations']:>9}")
    print("==================================================")
    
    # Quick sanity check
    assert gbm_metrics["max_drawdown"] < 0.35, "Warning: Maximum drawdown in stress testing exceeded baseline bounds!"
    print("Stress test validation check complete.")

if __name__ == "__main__":
    main()
