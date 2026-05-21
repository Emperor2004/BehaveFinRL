import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
import joblib

PROJECT_ROOT = Path("d:/Studies/DBM/BehaveFinRL")
sys.path.append(str(PROJECT_ROOT))
import config
from env.trading_env import TradingEnv
from regime.hmm import MarketRegimeDetector

class ValidationEarlyStoppingCallback(BaseCallback):
    """
    Custom Stable-Baselines3 callback for early stopping based on validation set performance.
    Evaluates the model every `eval_freq` steps. Stops training if the Sharpe ratio on
    the validation set does not improve for `patience` consecutive checks.
    """
    def __init__(self, val_df, regime_detector, eval_freq=2048, patience=5, verbose=1):
        super().__init__(verbose)
        self.val_df = val_df
        self.regime_detector = regime_detector
        self.eval_freq = eval_freq
        self.patience = patience
        
        self.best_sharpe = -np.inf
        self.patience_counter = 0
        self.eval_env = None
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            if self.eval_env is None:
                self.eval_env = TradingEnv(self.val_df, regime_detector=self.regime_detector)
            
            # Reset validation environment
            obs, info = self.eval_env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
            # Calculate validation Sharpe ratio
            val_net_worth = np.array(self.eval_env.net_worth_history)
            if len(val_net_worth) > 1:
                val_returns = np.diff(val_net_worth) / val_net_worth[:-1]
                mean_ret = np.mean(val_returns)
                std_ret = np.std(val_returns) if len(val_returns) > 1 else 1e-9
                val_sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(252)
            else:
                val_sharpe = -10.0
                
            if self.verbose > 0:
                print(f"[EarlyStopping] Step {self.num_timesteps} | Val Sharpe: {val_sharpe:.4f} (Best: {self.best_sharpe:.4f})")
                
            if val_sharpe > self.best_sharpe:
                self.best_sharpe = val_sharpe
                self.patience_counter = 0
                # Save best checkpoint
                save_path = PROJECT_ROOT / "saved_models" / f"best_model_seed_{self.model.seed}"
                self.model.save(save_path)
            else:
                self.patience_counter += 1
                if self.verbose > 0:
                    print(f"[EarlyStopping] Patience: {self.patience_counter}/{self.patience}")
                    
            if self.patience_counter >= self.patience:
                if self.verbose > 0:
                    print(f"[EarlyStopping] Early stopping triggered at step {self.num_timesteps}!")
                return False  # Stops training
                
        return True

def calculate_metrics(net_worth_history, benchmark_prices, holdings_history=None):
    """
    Computes financial performance and risk metrics.
    """
    net_worths = np.array(net_worth_history)
    portfolio_returns = np.diff(net_worths) / net_worths[:-1]
    
    # Cumulative return
    cum_return = (net_worths[-1] / net_worths[0]) - 1.0
    
    # Annualized Sharpe ratio (risk-free rate assumed = 0.0)
    if len(portfolio_returns) > 1 and portfolio_returns.std() > 0:
        sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
        
    # Annualized Sortino ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    if len(downside_returns) > 1 and downside_returns.std() > 0:
        sortino = (portfolio_returns.mean() / downside_returns.std()) * np.sqrt(252)
    else:
        sortino = sharpe
        
    # Maximum Drawdown
    peaks = np.maximum.accumulate(net_worths)
    drawdowns = (peaks - net_worths) / peaks
    max_dd = drawdowns.max()
    
    # Win rate (percent of active days with positive return)
    # Filter for active days: when holdings are non-zero (or absolute weight > 1%)
    if holdings_history is not None and len(holdings_history) == len(net_worths):
        active_days_mask = np.abs(np.array(holdings_history[:-1])) > 0.01
        active_returns = portfolio_returns[active_days_mask]
        if len(active_returns) > 0:
            win_rate = np.sum(active_returns > 0) / len(active_returns)
        else:
            win_rate = 0.0
    else:
        win_rate = np.sum(portfolio_returns > 0) / len(portfolio_returns) if len(portfolio_returns) > 0 else 0.0
        
    # Benchmark metrics
    bench_prices = np.array(benchmark_prices)
    bench_returns = np.diff(bench_prices) / bench_prices[:-1]
    bench_cum = (bench_prices[-1] / bench_prices[0]) - 1.0
    bench_sharpe = (bench_returns.mean() / bench_returns.std()) * np.sqrt(252) if len(bench_returns) > 1 and bench_returns.std() > 0 else 0.0
    bench_peaks = np.maximum.accumulate(bench_prices)
    bench_mdd = ((bench_peaks - bench_prices) / bench_peaks).max()
    
    return {
        "cumulative_return": cum_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "benchmark_return": bench_cum,
        "benchmark_sharpe": bench_sharpe,
        "benchmark_mdd": bench_mdd
    }

def train_ppo_agent(train_df, val_df, regime_detector, seed, config_module=config):
    """
    Trains a single PPO agent on the training env with a specific seed and early stopping callback.
    """
    print(f"\n--- Starting PPO Agent Training (Seed: {seed}) ---")
    
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)
    
    # Create training env
    train_env = TradingEnv(train_df, regime_detector=regime_detector)
    
    # Initialize PPO model with optimizer weight_decay (L2 regularization)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config_module.LEARNING_RATE,
        n_steps=config_module.N_STEPS,
        batch_size=config_module.BATCH_SIZE,
        gamma=config_module.GAMMA,
        gae_lambda=config_module.GAE_LAMBDA,
        clip_range=config_module.CLIP_RANGE,
        clip_range_vf=0.2,  # Stabilize value function updates against volatile financial rewards
        ent_coef=config_module.ENTROPY_COEF,
        verbose=1,
        tensorboard_log=None,
        seed=seed,
        policy_kwargs=dict(
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(weight_decay=1e-4)  # L2 Regularization (weight decay)
        )
    )
    
    # Create Early Stopping Callback
    es_callback = ValidationEarlyStoppingCallback(
        val_df=val_df,
        regime_detector=regime_detector,
        eval_freq=2048,
        patience=5,
        verbose=1
    )
    
    # Train PPO agent
    model.learn(total_timesteps=config_module.PPO_TIMESTEPS, callback=es_callback)
    
    # If early stopping saved a best model, load it before final save
    best_path = PROJECT_ROOT / "saved_models" / f"best_model_seed_{seed}.zip"
    if best_path.exists():
        print(f"Loading best checkpoint from validation early stopping for final model saving.")
        model = PPO.load(best_path, env=train_env)
        
    # Save the trained agent
    save_path = PROJECT_ROOT / "saved_models" / f"ppo_agent_seed_{seed}"
    model.save(save_path)
    print(f"PPO Agent (Seed {seed}) trained and saved to {save_path}.")
    
    return model

def evaluate_agent(model, test_df, regime_detector):
    """
    Evaluates a trained agent on the test DataFrame.
    """
    # Create test environment
    test_env = TradingEnv(test_df, regime_detector=regime_detector)
    obs, info = test_env.reset()
    
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
    # Extract results
    results = {
        "net_worth_history": test_env.net_worth_history,
        "holdings_history": test_env.holdings_history,
        "action_history": test_env.action_history,
        "regime_history": test_env.regime_history,
        "dates": test_df["date"].tolist(),
        "prices": test_df["adj_close"].tolist()
    }
    
    # Compute metrics
    metrics = calculate_metrics(results["net_worth_history"], results["prices"], results["holdings_history"])
    results["metrics"] = metrics
    
    return results

def run_multi_seed_training(quick_run=False):
    """
    Trains and evaluates PPO agents across all configured seeds.
    If quick_run is True, reduces timesteps for testing purposes.
    """
    # Load processed data
    train_path = PROJECT_ROOT / "data_cache" / f"{config.TICKER}_train.csv"
    test_path = PROJECT_ROOT / "data_cache" / f"{config.TICKER}_test.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Processed datasets not found. Run fetch.py and preprocess.py first.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Split train_df: use Lopez de Prado's Purged & Embargoed Cross-Validation (Fold 5)
    # to prevent data leakage and serial correlation overlap bias.
    from validation.cv import PurgedEmbargoKFold
    cv = PurgedEmbargoKFold(n_splits=5, purging_window=20, pct_embargo=0.01)
    splits = cv.split(train_df)
    train_idx, val_idx = splits[-1]  # Use Fold 5 for training/validation splits
    train_sub_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    
    # Load HMM regime detector
    hmm_path = PROJECT_ROOT / "saved_models" / "hmm_regime_detector.joblib"
    if not hmm_path.exists():
        raise FileNotFoundError("Trained HMM regime detector not found. Run regime/hmm.py first.")
        
    regime_detector = MarketRegimeDetector.load(hmm_path)
    
    # Override timesteps if quick_run is requested
    if quick_run:
        print("Performing a quick training run (dry run)...")
        # Store original timesteps
        orig_timesteps = config.PPO_TIMESTEPS
        config.PPO_TIMESTEPS = 2048  # Minimal training steps for compilation test
        
    evaluation_logs = {}
    
    for seed in config.SEEDS:
        model = train_ppo_agent(train_sub_df, val_df, regime_detector, seed)
        eval_results = evaluate_agent(model, test_df, regime_detector)
        evaluation_logs[seed] = eval_results
        
        m = eval_results["metrics"]
        print(f"Seed {seed} Evaluation Metrics:")
        print(f"  - Portfolio Return:  {m['cumulative_return']*100:+.2f}%")
        print(f"  - Portfolio Sharpe:  {m['sharpe_ratio']:.2f}")
        print(f"  - Portfolio Max DD: {m['max_drawdown']*100:.2f}%")
        print(f"  - Win Rate:          {m['win_rate']*100:.2f}%")
        print(f"  - Benchmark Return:  {m['benchmark_return']*100:+.2f}%")
        print(f"  - Benchmark Sharpe:  {m['benchmark_sharpe']:.2f}")
        
    # Restore original configuration if modified
    if quick_run:
        config.PPO_TIMESTEPS = orig_timesteps
        
    # Save evaluation logs
    log_path = PROJECT_ROOT / "logs" / "multi_seed_evaluations.joblib"
    joblib.dump(evaluation_logs, log_path)
    print(f"\nAll evaluations saved to {log_path}.")
    
    return evaluation_logs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a quick training run with 2048 steps")
    args = parser.parse_args()
    
    try:
        run_multi_seed_training(quick_run=args.quick)
    except Exception as e:
        print(f"Multi-seed training execution failed: {e}")
