# file: d:/Studies/DBM/BehaveFinRL/pipeline.py
"""Full BehaveFinRL pipeline orchestrator.

This script runs the end‑to‑end workflow:
  1️⃣ Fetch raw market data
  2️⃣ Pre‑process / feature engineer
  3️⃣ Hyper‑parameter optimisation (Optuna) – saves the best config and model
  4️⃣ SHAP explainability generation
  5️⃣ Stress‑testing (synthetic & historical)
  6️⃣ (Optional) launch the Flask dashboard for visualisation

All steps are executed via ``subprocess`` so they run in the same virtual environment.

Usage::
    python pipeline.py [--trials N]

    --trials N   Number of Optuna trials to run (default: 20)
    --no‑shap    Skip SHAP generation (useful for quick runs)
    --no‑stress  Skip stress‑testing
    --launch‑dash  Launch the Flask dashboard after the pipeline finishes
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd: str) -> None:
    """Execute *cmd* in a subprocess, printing the command first.
    If the command fails, the script aborts with the same exit code.
    """
    print(f"\n>>> {cmd}")
    # ``shell=True`` lets us pass a single string (e.g., "python script.py")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)

def main(trials: int = 20, shap: bool = True, stress: bool = True, launch_dash: bool = False) -> None:
    # ------------------------------------------------------------------
    # 1️⃣ Install / update dependencies (ensures a clean environment)
    # ------------------------------------------------------------------
    run_cmd("pip install -r requirements.txt")

    # ------------------------------------------------------------------
    # 2️⃣ Data download & preprocessing
    # ------------------------------------------------------------------
    run_cmd("python data/fetch.py")
    run_cmd("python data/preprocess.py")

    # ------------------------------------------------------------------
    # 3️⃣ Hyper‑parameter search with Optuna
    # ------------------------------------------------------------------
    run_cmd(f"python train.py --trials {trials}")

    # ------------------------------------------------------------------
    # 4️⃣ SHAP explainability (optional)
    # ------------------------------------------------------------------
    if shap:
        run_cmd("python explainability/shap_analysis.py")

    # ------------------------------------------------------------------
    # 5️⃣ Stress testing (optional)
    # ------------------------------------------------------------------
    if stress:
        run_cmd("python validation/stress_test.py")

    # ------------------------------------------------------------------
    # 6️⃣ Launch interactive dashboard (optional)
    # ------------------------------------------------------------------
    if launch_dash:
        print("\nLaunching Flask dashboard …")
        run_cmd("python dashboard/app.py")

    print("\n[OK]  Full pipeline completed.")
    # Show the saved best model location for quick reference
    best_models = list(Path("saved_models").glob("best_model_trial_*.zip"))
    if best_models:
        print("Best model saved at:", best_models[0])
    else:
        print("⚠️ No best model file found – check that the Optuna run succeeded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BehaveFinRL end‑to‑end pipeline")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials to run")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP explainability step")
    parser.add_argument("--no-stress", action="store_true", help="Skip stress testing step")
    parser.add_argument("--launch-dash", action="store_true", help="Start the Flask dashboard after the pipeline finishes")
    args = parser.parse_args()

    main(trials=args.trials, shap=not args.no_shap, stress=not args.no_stress, launch_dash=args.launch_dash)
