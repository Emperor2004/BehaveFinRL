# file: d:/Studies/DBM/BehaveFinRL/tune.py
"""Optuna hyper‑parameter optimisation for BehaveFinRL.
Runs a study, stores logs, checkpoints and final artefacts in the structured
folders defined in ``config.py``.
"""
import argparse
import os
import time
from pathlib import Path
from typing import Dict, Any

import optuna
import datetime
from optuna.pruners import MedianPruner, PatientPruner, NopPruner

from config import get_cfg, ROOT_DIR, STUDIES_DIR, LOGS_DIR, MODELS_DIR, BEST_CONFIG_DIR
from utils.logger import Logger
from utils.save_utils import dump_json, save_model
from train import train_one_trial


def _choose_pruner(name: str):
    if name == "median":
        return MedianPruner()
    if name == "patient":
        return PatientPruner()
    if name == "none":
        return NopPruner()


def main():
    parser = argparse.ArgumentParser(description="Run Optuna hyper‑parameter search for BehaveFinRL")
    parser.add_argument("--trials", type=int, default=None, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Overall timeout in minutes")
    parser.add_argument("--pruner", choices=["median", "patient", "none"], default="median")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs (processes)")
    args = parser.parse_args()

    # Build configuration – CLI args override defaults
    cfg = get_cfg()
    if args.trials:
        cfg["n_trials"] = args.trials
    if args.timeout:
        cfg["timeout_minutes"] = args.timeout
    cfg["pruner"] = args.pruner
    cfg["parallel_jobs"] = args.jobs

    # ------------------------------------------------------------------
    # Optuna study (SQLite) – new unique study each run
    # Create a fresh SQLite storage for this run to avoid compatibility issues
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    study_path = STUDIES_DIR / f"optuna_study_{timestamp}.db"
    study = optuna.create_study(
        study_name=f"behavefinrl_optuna_{timestamp}",
        storage=f"sqlite:///{study_path}",
        load_if_exists=False,
        direction="maximize",
        pruner=_choose_pruner(cfg["pruner"]),
    )

    # Logger for the whole tuning run
    main_logger = Logger(
        name="tuning",
        log_file=LOGS_DIR / "tuning.log",
        level=cfg["log_level"],
        tb_log_dir=ROOT_DIR / "runs" / "tuning",
    )
    main_logger.info(
        f"Starting Optuna study – trials={cfg['n_trials']}, jobs={cfg['parallel_jobs']}, pruner={cfg['pruner']}"
    )

    start_time = time.time()

    # ------------------------------------------------------------------
    # Objective wrapper – inject logger per trial
    # ------------------------------------------------------------------
    def objective(trial: optuna.trial.Trial):
        trial_logger = Logger(
            name=f"trial_{trial.number}",
            log_file=LOGS_DIR / f"trial_{trial.number}.log",
            level=cfg["log_level"],
            tb_log_dir=ROOT_DIR / "runs" / f"trial_{trial.number}",
        )
        try:
            model, final_reward = train_one_trial(trial, cfg, trial_logger)
            # Save checkpoint for this trial
            checkpoint_path = MODELS_DIR / f"model_trial_{trial.number}.zip"
            save_model(model, checkpoint_path)
            trial_logger.info("Checkpoint saved", path=str(checkpoint_path), reward=final_reward)
            return final_reward
        finally:
            trial_logger.close()

    # Run optimisation (supports resume)
    study.optimize(
        objective,
        n_trials=cfg["n_trials"],
        timeout=cfg["timeout_minutes"] * 60 if cfg["timeout_minutes"] else None,
        n_jobs=cfg["parallel_jobs"],
    )

    duration = time.time() - start_time
    best = study.best_trial
    main_logger.info(
        "Tuning finished",
        total_trials=len(study.trials),
        best_trial=best.number,
        best_reward=best.value,
        duration_seconds=duration,
    )

    # ------------------------------------------------------------------
    # Persist best hyper‑parameters and corresponding model
    # ------------------------------------------------------------------
    best_params_path = BEST_CONFIG_DIR / "best_params.json"
    dump_json(best.params, best_params_path)
    # Copy the best model checkpoint to a canonical location
    best_checkpoint_src = MODELS_DIR / f"model_trial_{best.number}.zip"
    best_model_path = MODELS_DIR / "best_model.zip"
    if best_checkpoint_src.exists():
        os.replace(best_checkpoint_src, best_model_path)
        main_logger.info("Best model saved", path=str(best_model_path))
    else:
        main_logger.warning("Best checkpoint missing – you may need to retrain the best config manually.")

    # ------------------------------------------------------------------
    # Visualisation – Optuna built‑in plots (saved as PNG)
    # ------------------------------------------------------------------
    try:
        import optuna.visualization as vis

        opt_hist = vis.plot_optimization_history(study)
        opt_hist.write_image(str(STUDIES_DIR / "optimization_history.png"))
        imp_plot = vis.plot_param_importances(study)
        imp_plot.write_image(str(STUDIES_DIR / "param_importances.png"))
        pc_plot = vis.plot_parallel_coordinate(study)
        pc_plot.write_image(str(STUDIES_DIR / "parallel_coordinate.png"))
        main_logger.info("Optuna visualisation plots saved.")
    except Exception as e:
        main_logger.error(f"Failed to generate Optuna plots: {e}")

    main_logger.close()


if __name__ == "__main__":
    main()
