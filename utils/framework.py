import os
import importlib
from typing import Tuple, Any, Dict


def detect_rl_framework() -> str:
    """Detect which RL framework is installed in the environment.
    Returns one of: 'stable_baselines3', 'cleanrl', 'rllib', or raises ImportError.
    """
    # Order matters – prefer more feature‑rich libraries.
    candidates = [
        ("stable_baselines3", "stable_baselines3"),
        ("cleanrl", "cleanrl"),
        ("ray.rllib", "ray.rllib"),
    ]
    for name, import_path in candidates:
        try:
            importlib.import_module(import_path)
            return name
        except ImportError:
            continue
    raise ImportError("No supported RL framework (stable-baselines3, cleanrl, rllib) found in the environment.")


def get_algorithm_class(framework: str, algorithm_name: str):
    """Return the class implementing ``algorithm_name`` for the detected framework.
    Supported algorithms (as of now):
        - PPO (Stable‑Baselines3, CleanRL, RLlib)
        - A2C, DDPG, SAC, etc. can be added later.
    """
    if framework == "stable_baselines3":
        module = importlib.import_module("stable_baselines3")
        return getattr(module, algorithm_name)
    if framework == "cleanrl":
        # CleanRL uses functional API; we wrap a simple stub for compatibility.
        raise NotImplementedError("CleanRL integration not implemented yet.")
    if framework == "rllib":
        # RLlib uses Trainer classes; map common names.
        from ray.rllib.agents import ppo as rllib_ppo
        mapping = {"PPO": rllib_ppo.PPOTrainer}
        if algorithm_name in mapping:
            return mapping[algorithm_name]
        raise ValueError(f"Algorithm {algorithm_name} not supported for RLlib.")
    raise ValueError(f"Unsupported framework {framework}")
