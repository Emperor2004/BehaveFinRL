# file: d:/Studies/DBM/BehaveFinRL/train.py
"""Training script used by Optuna trials.
Encapsulates a single PPO training run, model creation with sampled
hyper‑parameters, and evaluation.
"""
import random
from typing import Any, Dict, Tuple
import shutil

import numpy as np
import pandas as pd
import torch
import optuna
import datetime
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import get_cfg, ROOT_DIR
from env.trading_env import TradingEnv
from regime.hmm import MarketRegimeDetector
from utils.logger import Logger
from utils.save_utils import save_model


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducibility across random, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre‑processed training and test CSVs and return them."""
    train_path = ROOT_DIR / "data_cache" / f"{get_cfg()['TICKER']}_train.csv"
    test_path = ROOT_DIR / "data_cache" / f"{get_cfg()['TICKER']}_test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Training/test CSVs missing – run data fetching scripts first.")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def make_env(df: pd.DataFrame, regime_detector: MarketRegimeDetector) -> DummyVecEnv:
    """Wrap :class:`TradingEnv` in a vectorised environment expected by SB3."""
    env = TradingEnv(df, regime_detector=regime_detector)
    return DummyVecEnv([lambda: env])


def evaluate(
    model: PPO,
    df: pd.DataFrame,
    regime_detector: MarketRegimeDetector,
    cfg: Dict[str, Any],
    episodes: int = 10,
) -> float:
    """Run *episodes* episodes and return mean net‑worth return.
    Metric: (final_worth - initial_balance) / initial_balance.
    """
    env = make_env(df, regime_detector)
    returns = []
    for _ in range(episodes):
        # Support both single output and (obs, info) tuple from reset()
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, _, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # Older gym API returns (obs, reward, done, info)
                obs, _, done, info = step_result
                # No separate truncated flag; treat as not truncated

        # info is a dict with net_worth key
        # info may be a list of dicts (vectorized env); extract the dict
        if isinstance(info, list):
            info = info[0]
        final_worth = info["net_worth"]
        returns.append((final_worth - cfg["INITIAL_BALANCE"]) / cfg["INITIAL_BALANCE"])
    env.close()
    return float(np.mean(returns))


def train_one_trial(
    trial: optuna.trial.Trial,
    cfg_overrides: Dict[str, Any],
    logger: Logger,
) -> Tuple[PPO, float]:
    """Run a single Optuna trial, returning the trained model and its reward."""
    # ---------- Hyper‑parameter sampling ----------
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.80, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 1e-5, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    try:
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    except Exception:
        weight_decay = 0.0
    batch_size = trial.suggest_categorical("batch_size", (256, 512, 1024))
    n_steps = trial.suggest_categorical("n_steps", (1024, 2048, 4096))
    activation_fn = trial.suggest_categorical("activation_fn", ("relu", "tanh"))
    arch_idx = trial.suggest_int("net_arch_idx", 0, 2)
    arch_options = [(64, 64), (128, 128), (256, 256)]
    net_arch = list(arch_options[arch_idx])

    cfg = get_cfg(**cfg_overrides)
    set_global_seed(cfg["seed"])

    # ---------- Load data & HMM ----------
    train_df, test_df = load_data()
    # ---------- Load HMM detector ----------
    from regime.hmm import MarketRegimeDetector as HMMCls
    hmm_path = ROOT_DIR / "saved_models" / "hmm_regime_detector.joblib"
    if not hmm_path.exists():
        logger.info("HMM detector not found, training now.")
        regime_detector = HMMCls()
        regime_detector.fit(train_df)
        regime_detector.save(hmm_path)
        logger.info("HMM detector trained and saved.")
    else:
        try:
            regime_detector = HMMCls.load(hmm_path)
        except Exception as e:
            logger.warning(f"Failed to load HMM detector ({e}), retraining.")
            regime_detector = HMMCls()
            regime_detector.fit(train_df)
            regime_detector.save(hmm_path)
            logger.info("Re‑trained HMM detector and saved for future trials.")

    # ---------- Environment ----------
    env = make_env(train_df, regime_detector)

    # ---------- Model ----------
    # Map activation function string to torch class
    activation_map = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}
    policy_kwargs = {
        "activation_fn": activation_map[activation_fn.lower()],
        "net_arch": net_arch,
    }
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        batch_size=batch_size,
        n_steps=n_steps,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=cfg["device"],
        seed=cfg["seed"],
    )
    # Apply L2 regularization via weight decay by resetting optimizer
    model.policy.optimizer = torch.optim.Adam(
        model.policy.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # ---------- Training loop with intermediate reporting ----------
    total_timesteps = cfg.get("PPO_TIMESTEPS", 50000)
    report_interval = max(total_timesteps // 10, 1_000)
    elapsed = 0
    while elapsed < total_timesteps:
        step = min(report_interval, total_timesteps - elapsed)
        # Use a positive log_interval to avoid ZeroDivisionError
        model.learn(total_timesteps=step, reset_num_timesteps=False, log_interval=10)
        elapsed += step
        interm_reward = evaluate(model, test_df, regime_detector, cfg, episodes=3)
        trial.report(interm_reward, elapsed)
        logger.info(
            f"Trial {trial.number} – step {elapsed}/{total_timesteps} – interm reward {interm_reward:.4f}",
            step=elapsed,
            reward=interm_reward,
        )
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Final evaluation
    final_reward = evaluate(model, test_df, regime_detector, cfg, episodes=cfg.get("eval_episodes", 10))
    return model, final_reward

# ------------------------------------------------------------------
# Simple sanity‑check entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1, help="Number of quick trials to run")
    parser.add_argument("--load-best", action="store_true", help="Load hyper‑parameters from best_hyperparams.json as defaults")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load saved hyper‑parameters if requested
    # ------------------------------------------------------------------
    cfg_overrides = {}
    if args.load_best:
        best_path = Path(__file__).parent / "best_hyperparams.json"
        if best_path.is_file():
            with best_path.open("r", encoding="utf-8") as f:
                cfg_overrides = json.load(f)
            print(f"🔧 Loaded best hyper‑parameters from {best_path}")
        else:
            print("⚠️ No best_hyperparams.json found – running with fresh sampling")

    study = optuna.create_study(direction="maximize", storage="sqlite:///behavefinrl_optuna.db", study_name="behavefinrl_study", load_if_exists=True)
    cfg = get_cfg()
    logger = Logger("standalone", log_file=ROOT_DIR / "logs" / "standalone.log")

    def obj(trial):
        model, reward = train_one_trial(trial, cfg_overrides, logger)
        return reward

    study.optimize(obj, n_trials=args.trials)

    # ------------------------------------------------------------------
    # Save the best hyper‑parameters for future runs
    # ------------------------------------------------------------------
    best_params = study.best_trial.params

    best_path = Path(__file__).parent / "best_hyperparams.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)
    print(f"Saved best hyper-parameters to {best_path}")
    # ------------------------------------------------------------------
    # Re‑train the best trial's model and save it
    # ------------------------------------------------------------------
    class _BestTrial:
        def __init__(self, params):
            # Preserve all parameters; weight_decay may be absent
            self.params = params
        def suggest_float(self, name, low, high, log=False):
            # Return the sampled value if present, otherwise default to 0.0
            return self.params.get(name, 0.0)
        def suggest_categorical(self, name, choices):
            return self.params[name]
        def suggest_int(self, name, low, high):
            return self.params[name]
        def report(self, *a, **kw):
            pass
        def should_prune(self):
            return False
        @property
        def number(self):
            return 0

# Re‑train the best trial's model and save it
best_trial_obj = _BestTrial(best_params)
best_model, _ = train_one_trial(best_trial_obj, {}, logger)

# Save the best model with a deterministic filename (timestamp + study UUID) and create a stable symlink
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
study_uuid = str(study._study_id)[:8]  # short identifier for readability
model_filename = f"{timestamp}-Study-{study_uuid}_best_model_trial_{study.best_trial.number}.zip"
model_path = ROOT_DIR / "saved_models" / model_filename
save_model(best_model, model_path)
print(f"Saved best model to {model_path}")
# Update (or create) a symlink called latest_best_model.zip that always points to the newest best model.
latest_link = ROOT_DIR / "saved_models" / "latest_best_model.zip"
try:
    latest_link.unlink()
except FileNotFoundError:
    pass
# Try to create a symlink; if insufficient privileges, fall back to copying the file
try:
    latest_link.symlink_to(model_filename)
    print(f"Created symlink: {latest_link} -> {model_filename}")
except OSError as e:
    # Windows may raise a privilege error; copy the model file instead
    target_path = ROOT_DIR / "saved_models" / model_filename
    shutil.copyfile(target_path, latest_link)
    print(f"Symlink failed ({e}); copied model to {latest_link} instead.")

print("Best trial", study.best_trial.number, "reward", study.best_trial.value)
