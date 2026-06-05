import os
import random
import logging

# Default logging level
LOG_LEVEL = logging.INFO

def get_cfg(**overrides) -> dict:
    """Return configuration dictionary.

    Starts from module-level defaults and applies any overrides passed
    via keyword arguments.
    """
    cfg = {
        "TICKER": TICKER,
        "START_DATE": START_DATE,
        "END_DATE": END_DATE,
        "PROSPECT_ALPHA": PROSPECT_ALPHA,
        "PROSPECT_BETA": PROSPECT_BETA,
        "PROSPECT_LAMBDA": PROSPECT_LAMBDA,
        "PROB_WEIGHT_GAMMA": PROB_WEIGHT_GAMMA,
        "LAMBDA_BULL": LAMBDA_BULL,
        "LAMBDA_BEAR": LAMBDA_BEAR,
        "LAMBDA_VOLATILE": LAMBDA_VOLATILE,
        "HMM_STATES": HMM_STATES,
        "HMM_COVARIANCE_TYPE": HMM_COVARIANCE_TYPE,
        "HMM_RANDOM_STATE": HMM_RANDOM_STATE,
        "PPO_TIMESTEPS": PPO_TIMESTEPS,
        "LEARNING_RATE": LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "N_STEPS": N_STEPS,
        "GAMMA": GAMMA,
        "ENTROPY_COEF": ENTROPY_COEF,
        "GAE_LAMBDA": GAE_LAMBDA,
        "CLIP_RANGE": CLIP_RANGE,
        "SEEDS": SEEDS,
        "INITIAL_BALANCE": INITIAL_BALANCE,
        "TRANSACTION_FEE_PCT": TRANSACTION_FEE_PCT,
        "SLIPPAGE_COEF": SLIPPAGE_COEF,
        "MARKET_IMPACT_COEF": MARKET_IMPACT_COEF,
        "SORTINO_WINDOW": SORTINO_WINDOW,
        "MAX_DRAWDOWN_LIMIT": MAX_DRAWDOWN_LIMIT,
        "VAR_ALPHA": VAR_ALPHA,
        "VAR_LIMIT": VAR_LIMIT,
        "VAR_PENALTY_COEF": VAR_PENALTY_COEF,
        "DD_PENALTY_COEF": DD_PENALTY_COEF,
        "ACTION_REG_COEF": ACTION_REG_COEF,
        "POSITION_CAP_BULL": POSITION_CAP_BULL,
        "POSITION_CAP_VOLATILE": POSITION_CAP_VOLATILE,
        "POSITION_CAP_BEAR": POSITION_CAP_BEAR,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": random.randint(0, 2**32 - 1),
        "log_level": LOG_LEVEL,
        # tuning defaults
        "n_trials": 50,
        "timeout_minutes": None,
        "pruner": "median",
        "parallel_jobs": 1,
        "eval_episodes": 10,
    }
    cfg.update(overrides)
    return cfg

from pathlib import Path
from dotenv import load_dotenv
import logging
import torch

# Load environment variables from .env
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_cache"
MODEL_DIR = BASE_DIR / "saved_models"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Additional path constants used throughout the project
ROOT_DIR = BASE_DIR
STUDIES_DIR = ROOT_DIR / "studies"
STUDIES_DIR.mkdir(exist_ok=True)
BEST_CONFIG_DIR = ROOT_DIR / "best_configs"
BEST_CONFIG_DIR.mkdir(exist_ok=True)
BEST_CONFIG_DIR.mkdir(exist_ok=True)
LOGS_DIR = LOG_DIR
MODELS_DIR = MODEL_DIR
BEST_CONFIG_DIR = ROOT_DIR / "best_configs"

# API Keys
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Asset & Date Configurations
TICKER = "SP500"  # We use the S&P 500 Index from FRED to ensure free daily history
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"

# Macroeconomic indicators from FRED (Series IDs)
# T10Y2Y: 10-Year minus 2-Year Treasury yield spread (yield curve)
# DFF: Effective Federal Funds Rate (central bank interest rate policy)
# VIXCLS: CBOE Volatility Index (daily close)
FRED_SERIES = {
    "yield_spread": "T10Y2Y",
    "fed_rate": "DFF",
    "vix": "VIXCLS"
}

# Prospect Theory Parameters (Tversky & Kahneman, 1992)
PROSPECT_ALPHA = 0.88   # Diminishing sensitivity exponent for gains
PROSPECT_BETA = 0.88    # Diminishing sensitivity exponent for losses
PROSPECT_LAMBDA = 2.25  # Baseline loss aversion coefficient

# Probability Weighting (Tversky-Kahneman / Prelec)
PROB_WEIGHT_GAMMA = 0.65  # Curvature parameter γ for probability weighting functions

# Regime-Adaptive Loss Aversion Parameters
# HMM Regime mapping: Bull (state 0), Bear (state 1), High Volatility (state 2)
LAMBDA_BULL = 2.00      # Slightly relaxed loss aversion in bull markets
LAMBDA_BEAR = 2.75      # Increased loss aversion in bear markets to protect capital
LAMBDA_VOLATILE = 2.40  # Moderately increased loss aversion in high volatility

# Hidden Markov Model (HMM) Parameters
HMM_STATES = 3
HMM_COVARIANCE_TYPE = "diag"
HMM_RANDOM_STATE = 42

# Reinforcement Learning (PPO) Hyperparameters
PPO_TIMESTEPS = 50000
LEARNING_RATE = 3e-4
BATCH_SIZE = 512        # Larger mini-batch size to smooth out non-stationarity
N_STEPS = 2048
GAMMA = 0.99
ENTROPY_COEF = 0.01
GAE_LAMBDA = 0.92       # Tightly tuned between 0.90 and 0.95
CLIP_RANGE = 0.2

# Multi-seed evaluation configurations
SEEDS = [42, 7, 21, 99, 5]

# Environment & Microstructure Friction Configuration
INITIAL_BALANCE = 10000.0
TRANSACTION_FEE_PCT = 0.0015  # 0.15% base fee per transaction
SLIPPAGE_COEF = 0.05          # Slippage factor scaled by rolling volatility
MARKET_IMPACT_COEF = 0.002    # Quadratic market impact penalty (was 0.02 — caused reward collapse)

# Risk Constraints & Reward shaping
SORTINO_WINDOW = 63           # 3-month rolling window for Sortino calculation
MAX_DRAWDOWN_LIMIT = 0.15     # 15% Max Drawdown Constraint
VAR_ALPHA = 0.95              # 95% Value-at-Risk confidence
VAR_LIMIT = 0.03              # 3% maximum Value-at-Risk constraint
VAR_PENALTY_COEF = 10.0       # Penalty multiplier for VaR breach (was 50.0 — dominated reward)
DD_PENALTY_COEF = 10.0        # Penalty multiplier for Drawdown breach (was 50.0 — dominated reward)
ACTION_REG_COEF = 0.005       # Penalty for target weight changes (was 0.5 — the PRIMARY cause of collapse)

# Regime-Scaled Position Sizing Caps
POSITION_CAP_BULL     = 1.0   # Bull regime: full ±1.0 range allowed
POSITION_CAP_VOLATILE = 0.75  # Volatile regime: shrink to ±0.75
POSITION_CAP_BEAR     = 0.6   # Bear regime: shrink to ±0.6
