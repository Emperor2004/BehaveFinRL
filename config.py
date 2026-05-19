import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_cache"
MODEL_DIR = BASE_DIR / "saved_models"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they do not exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

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
MARKET_IMPACT_COEF = 0.02     # Quadratic market impact penalty

# Risk Constraints & Reward shaping
SORTINO_WINDOW = 63           # 3-month rolling window for Sortino calculation
MAX_DRAWDOWN_LIMIT = 0.15     # 15% Max Drawdown Constraint
VAR_ALPHA = 0.95              # 95% Value-at-Risk confidence
VAR_LIMIT = 0.03              # 3% maximum Value-at-Risk constraint
VAR_PENALTY_COEF = 50.0       # Penalty multiplier for VaR breach
DD_PENALTY_COEF = 50.0        # Penalty multiplier for Drawdown breach
ACTION_REG_COEF = 0.5         # Penalty multiplier for target weight changes (regularization)

