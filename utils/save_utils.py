# file: d:/Studies/DBM/BehaveFinRL/utils/save_utils.py
"""Helper utilities for persisting models, checkpoints and JSON artefacts.
All functions create missing parent directories automatically.
"""

import json
from pathlib import Path
from typing import Any, Dict

from stable_baselines3.common.base_class import BaseAlgorithm


def save_model(model: BaseAlgorithm, path: Path) -> None:
    """Save a Stable‑Baselines3 model to ``path`` (creates parent directories)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def load_model(path: Path) -> BaseAlgorithm:
    """Load a Stable‑Baselines3 model from ``path`` (expects a .zip file)."""
    from stable_baselines3 import PPO  # generic import; actual class stored in zip

    return PPO.load(str(path))


def dump_json(obj: Any, path: Path, *, indent: int = 2) -> None:
    """Write *obj* (must be JSON‑serialisable) to *path*.
    Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file from *path* and return its contents as a dict."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
