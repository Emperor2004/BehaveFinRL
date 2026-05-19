import sys
import joblib
import numpy as np
import pandas as pd
import torch
import shap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to save plots without UI
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("d:/Studies/DBM/BehaveFinRL")
sys.path.append(str(PROJECT_ROOT))

import config
from env.trading_env import TradingEnv
from regime.hmm import MarketRegimeDetector
from stable_baselines3 import PPO

def predict_action_weight(model, obs_array):
    """
    Wraps the PPO policy forward pass to return the continuous action (target weight)
    for a given numpy array of observations. Compatible with SHAP KernelExplainer.
    """
    if len(obs_array.shape) == 1:
        obs_array = np.expand_dims(obs_array, axis=0)
        
    obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(model.policy.device)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        actions = distribution.mode()
    return actions.cpu().numpy().squeeze(-1)

def run_shap_analysis():
    """
    Executes SHAP explainability analysis on the trained PPO agent.
    Computes SHAP feature importance values per market regime and saves plots/metrics.
    """
    print("\n--- Starting SHAP Policy Explainability Analysis ---")
    
    # Load test data and models
    train_path = PROJECT_ROOT / "data_cache" / f"{config.TICKER}_train.csv"
    test_path = PROJECT_ROOT / "data_cache" / f"{config.TICKER}_test.csv"
    hmm_path = PROJECT_ROOT / "saved_models" / "hmm_regime_detector.joblib"
    
    # Load first seed agent (Seed 42) as the primary model to analyze
    agent_path = PROJECT_ROOT / "saved_models" / "ppo_agent_seed_42.zip"
    
    if not train_path.exists() or not test_path.exists() or not hmm_path.exists() or not agent_path.exists():
        # Fall back to any available zip file in saved_models
        zip_files = list((PROJECT_ROOT / "saved_models").glob("*.zip"))
        if len(zip_files) == 0:
            raise FileNotFoundError("No trained PPO models found. Run training first.")
        agent_path = zip_files[0]
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    regime_detector = MarketRegimeDetector.load(hmm_path)
    model = PPO.load(agent_path)
    
    # Initialize trading envs to get observation matrices
    train_env = TradingEnv(train_df, regime_detector=regime_detector)
    test_env = TradingEnv(test_df, regime_detector=regime_detector)
    
    # Collect observations using continuous dummy action
    dummy_action = np.array([0.0], dtype=np.float32)
    
    train_obs = []
    obs, _ = train_env.reset()
    train_obs.append(obs)
    for _ in range(len(train_df) - 2):
        obs, _, _, _, _ = train_env.step(dummy_action)
        train_obs.append(obs)
    train_obs = np.array(train_obs)
    
    test_obs = []
    test_regimes = []
    obs, _ = test_env.reset()
    test_obs.append(obs)
    test_regimes.append(test_env.regimes[0])
    for i in range(1, len(test_df) - 1):
        obs, _, _, _, _ = test_env.step(dummy_action)
        test_obs.append(obs)
        test_regimes.append(test_env.regimes[i])
    test_obs = np.array(test_obs)
    test_regimes = np.array(test_regimes)
    
    # Feature Names
    # Note: TradingEnv appends portfolio weight and last return at the end of feature_cols
    feature_names = train_env.feature_cols + ["portfolio_weight", "last_step_return"]
    
    # Establish background dataset for SHAP (k-means to keep calculation fast)
    background = shap.kmeans(train_obs, 20)
    
    # Define custom predict wrapper for continuous target weight
    predict_fn = lambda x: predict_action_weight(model, x)
    
    # Initialize KernelExplainer
    explainer = shap.KernelExplainer(predict_fn, background)
    
    # We will analyze and plot SHAP values separately for each of the 3 market regimes
    regime_labels = {0: "Bull", 1: "Bear", 2: "Volatile"}
    shap_results = {}
    
    # Create output directory for SHAP assets
    shap_assets_dir = PROJECT_ROOT / "dashboard" / "static" / "images"
    shap_assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Set plot styling (pure matplotlib)
    plt.rcParams["figure.facecolor"] = "#1e1e1e"
    plt.rcParams["axes.facecolor"] = "#2d2d2d"
    plt.rcParams["text.color"] = "#ffffff"
    plt.rcParams["axes.labelcolor"] = "#ffffff"
    plt.rcParams["xtick.color"] = "#ffffff"
    plt.rcParams["ytick.color"] = "#ffffff"
    plt.rcParams["grid.color"] = "#3d3d3d"
    
    for r_state, r_name in regime_labels.items():
        print(f"Running SHAP analysis for {r_name} Regime...")
        # Get observations in this regime
        r_indices = np.where(test_regimes == r_state)[0]
        obs_source = test_obs
        
        if len(r_indices) == 0:
            print(f"No test observations found for {r_name} regime. Falling back to training dataset...")
            train_regimes = train_env.regimes[:len(train_obs)]
            r_indices = np.where(train_regimes == r_state)[0]
            obs_source = train_obs
            
        if len(r_indices) == 0:
            print(f"No observations found in either test or train set for {r_name} regime. Skipping.")
            continue
            
        # Sample up to 25 observations from this regime to run SHAP on
        np.random.seed(42)
        sample_indices = np.random.choice(r_indices, size=min(25, len(r_indices)), replace=False)
        r_obs_sample = obs_source[sample_indices]
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(r_obs_sample, nsamples=100)
        
        # Calculate average absolute SHAP values (feature importance)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Save key importance metrics
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)
        
        shap_results[r_name] = importance_df.to_dict(orient="records")
        
        # Generate and save horizontal bar chart
        plt.figure(figsize=(10, 6))
        
        color_map = {"Bull": "#10b981", "Bear": "#ef4444", "Volatile": "#f59e0b"}
        bar_color = color_map.get(r_name, "#3b82f6")
        
        top_n = min(10, len(importance_df))
        top_features = importance_df.head(top_n).iloc[::-1]
        
        plt.barh(
            top_features["feature"],
            top_features["mean_abs_shap"],
            color=bar_color,
            edgecolor="none"
        )
        plt.grid(True, axis="x", linestyle="--", alpha=0.3)
        
        plt.title(f"SHAP Policy Feature Attribution — {r_name} Regime (Action: Weight)", fontsize=14, color="#ffffff", pad=15)
        plt.xlabel("Mean Absolute SHAP Value (Impact on Weight Decision)", fontsize=12, color="#ffffff")
        plt.ylabel("Market / Portfolio Feature", fontsize=12, color="#ffffff")
        plt.tight_layout()
        
        # Save image
        img_path = shap_assets_dir / f"shap_importance_{r_name.lower()}.png"
        plt.savefig(img_path, dpi=150, facecolor="#1e1e1e")
        plt.close()
        print(f"SHAP plot for {r_name} saved to {img_path}.")
        
    # Save the computed SHAP values to JSON/joblib for Flask dashboard access
    shap_data_path = PROJECT_ROOT / "logs" / "shap_values.joblib"
    joblib.dump(shap_results, shap_data_path)
    print(f"SHAP feature importance stats saved to {shap_data_path}.")
    print("SHAP explainability pipeline finished.")

if __name__ == "__main__":
    try:
        run_shap_analysis()
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
