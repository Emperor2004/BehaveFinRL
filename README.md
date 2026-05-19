# BehaveFinRL

> A Prospect Theory-Driven Reinforcement Learning Trading Agent  
> with Explainable, Regime-Adaptive Decision Making under Continuous CMDP Constraints

---

## Overview

BehaveFinRL is a research-grade algorithmic trading system that encodes **Behavioural Finance theory** directly into a Reinforcement Learning framework. Rather than training an agent to maximise raw expected return — the standard assumption of rational actor models — BehaveFinRL trains an agent whose reward structure reflects how human investors actually experience financial outcomes.

The core insight comes from **Prospect Theory (Kahneman & Tversky, 1979)**: people do not evaluate outcomes in absolute terms. They evaluate them relative to a reference point, and they feel losses approximately 2.25× more intensely than equivalent gains. BehaveFinRL encodes this asymmetry into the agent's reward function.

To ensure realistic backtesting and simulation boundaries, this framework treats the environment as a **Constrained Markov Decision Process (CMDP)** with continuous portfolio allocations, real market microstructures (spreads, slippage, and impact costs), out-of-sample early stopping, policy regularization, and rigorous synthetic stress testing.

---

## Key Features

- **Continuous Portfolio Allocations** — Upgraded the agent to continuous target weights $a_t \in [-1, 1]$ (representing long, short, or neutral weight allocations) via a Gaussian policy.
- **Prospect Theory Reward Shaping** — Custom value function replaces raw return with a psychologically grounded reward ($λ=2.25$, $α=β=0.88$).
- **CMDP Risk Constraints** — Implemented quadratic penalty boundary scaling for 95% Value-at-Risk (VaR) and Maximum Drawdown breaches.
- **Frictional Drag Modeling** — Simulates linear broker fees, bid-ask spreads scaled by rolling volatility, and quadratic market impact costs to prevent unrealistically profitable backtests.
- **Market Regime Detection (HMM)** — Hidden Markov Model classifies market states into Bull, Bear, and High Volatility regimes to dynamically scale the loss aversion coefficient $\lambda$.
- **Validation-Based Early Stopping & Regularization** — Monitors out-of-sample validation Sharpe ratios to abort training early, paired with L2 optimizer weight decay and action smoothing penalties to prevent over-trading.
- **Purged & Embargoed Cross-Validation** — Implements Lopez de Prado's cross-validation splits to neutralize serial correlation and overlap bias.
- **Stress-Testing Engine** — Evaluates trained policies on synthetic Geometric Brownian Motion (GBM) paths, GARCH(1,1) volatility clustering, and historical bear market sub-periods.
- **Policy Explainability (SHAP)** — Employs KernelSHAP to attribute continuous target weight allocations to features across HMM regimes.
- **Flask Dashboard** — A premium dark-themed web interface for visualising training metrics, regime timelines, equity curves, SHAP plots, and risk metrics.

---

## Project Structure

```
BehaveFinRL/
│
├── data/
│   ├── fetch.py              # Alpha Vantage (OHLCV) + FRED (macro indicators) API calls
│   └── preprocess.py         # Normalisation, feature engineering, and microstructure proxy generation
│
├── env/
│   └── trading_env.py        # Custom Gymnasium environment with continuous weights and CMDP penalties
│
├── models/
│   ├── reward.py             # Prospect Theory value function
│   └── agent.py              # PPO setup, validation early stopping callback, and training loop
│
├── regime/
│   └── hmm.py                # HMM training, regime labelling, regime-λ mapping
│
├── explainability/
│   └── shap_analysis.py      # KernelSHAP pipeline for continuous weight attribution
│
├── validation/
│   ├── cv.py                 # Lopez de Prado's Purged & Embargoed Cross-Validation
│   └── stress_test.py        # Synthetic (GBM, GARCH) and historical stress testing
│
├── dashboard/
│   ├── app.py                # Flask application entry point
│   └── templates/            # HTML templates (dark theme)
│       ├── index.html
│       ├── training.html
│       ├── regime.html
│       └── shap.html
│
├── tests/
│   ├── run_all.py            # Main unit test suite runner
│   ├── test_cv.py            # Purged & Embargoed CV validation tests
│   ├── test_env.py           # Gymnasium continuous action and transaction cost tests
│   ├── test_reward.py        # Prospect Theory reward calculations
│   └── test_hmm.py           # HMM fitting and state mapping tests
│
├── config.py                 # Hyperparameters, risk limits, and API keys
├── train.py                  # Unified training pipeline runner
├── .env.example              # Template environment variables file
├── .gitignore                # Upgraded project version ignore paths
├── requirements.txt          # Python dependency list
└── README.md                 # Project documentation
```

---

## Theoretical & Mathematical Background

### 1. Prospect Theory Value Function

The reward function replaces raw portfolio return $r_t$ with a prospect-theoretic utility value $v(r_t)$:

$$v(x) = \begin{cases} 
      x^\alpha & \text{if } x \geq 0 \quad (\text{gains — concave, diminishing sensitivity}) \\
      -\lambda (-x)^\beta & \text{if } x < 0 \quad (\text{losses — convex, loss aversion}) 
   \end{cases}$$

**Baseline Exponents**: $\alpha = \beta = 0.88$, $\lambda = 2.25$.

### 2. Regime-Adaptive $\lambda$ Scaling

| HMM Regime | $\lambda$ Adjustment | Agent Behaviour |
|---|---|---|
| **Bull** | Slightly relaxed ($\lambda = 2.00$) | Permissive trend-following and long weights |
| **Bear** | Scaled upward ($\lambda = 2.75$) | Rapid liquidation to cash / conservative posture |
| **High Volatility** | Moderate increase ($\lambda = 2.40$) | Size reduction and volatility-conscious trading |

### 3. CMDP Penalties & Action Regularization

To prevent drawdowns and high-frequency action chattering under high market frictions:

$$\text{Reward}_{\text{shaped}} = v(R_t) - C_{\text{VaR}} - C_{\text{Drawdown}} - C_{\text{Action}}$$

Where:
- **Value-at-Risk Penalty**: $C_{\text{VaR}} = \text{VAR\_PENALTY\_COEF} \cdot \max(0, \text{VaR}_{95\%} - \text{VAR\_LIMIT})^2$
- **Drawdown Penalty**: $C_{\text{Drawdown}} = \text{DD\_PENALTY\_COEF} \cdot \max(0, \text{MaxDD} - \text{MAX\_DRAWDOWN\_LIMIT})^2$
- **Action Regularization**: $C_{\text{Action}} = \text{ACTION\_REG\_COEF} \cdot (a_t - a_{t-1})^2$

---

## Installation & Setup

### Prerequisites
- Python 3.9+

### Setup
```bash
# Clone the repository
git clone https://github.com/omnarayanpandit/behaverl.git
cd behaverl

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Linux/macOS: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration
Copy the sample environment variables file and fill in your API credentials:
```bash
cp .env.example .env
```
Open `.env` and configure your keys:
- `ALPHA_VANTAGE_KEY`
- `FRED_API_KEY`

---

## Usage

### 1. Unified Training Pipeline
Run the unified training script to fetch data, preprocess features, fit the HMM regime detector, train the regularized PPO agent (across 5 seeds), and evaluate results:
```bash
python train.py
```
*To run a quick dry-run with reduced timesteps (2048 steps per seed) to verify compilation:*
```bash
python train.py --quick
```

### 2. Generate SHAP Explanations
Calculate feature attribution charts across HMM regimes for the trained continuous policy:
```bash
python explainability/shap_analysis.py
```

### 3. Run Stress Testing
Execute synthetic GBM/GARCH simulations and historical stress evaluations:
```bash
python validation/stress_test.py
```

### 4. Launch Flask Dashboard
Start the interactive visualization server:
```bash
python dashboard/app.py
```
Open your browser and navigate to: **http://127.0.0.1:5000**

---

## Testing

Execute the unit test suite to verify pipeline integrity:
```bash
python tests/run_all.py
```

---

## References

- Kahneman, D., & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. *Econometrica*, 47(2), 263–291.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- Raffin, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *JMLR*, 22(268).
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.

---

## Author

**Om Narayan Pandit**  
Symbiosis Institute of Business Management, Pune  
GitHub: [@Emperor2004](https://github.com/Emperor2004)  
Portfolio: [om-narayan-pandit.vercel.app](https://om-narayan-pandit.vercel.app)

*Disclaimer: Academic research project. Not investment advice.*