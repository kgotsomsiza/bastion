from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("config/models.json")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if os.getenv("FRUGAL_REMOTE_THRESHOLD"):
        config["remote_threshold"] = float(os.environ["FRUGAL_REMOTE_THRESHOLD"])

    if os.getenv("FIREWORKS_MODEL_ID"):
        config.setdefault("fireworks", {})["model_id"] = os.environ["FIREWORKS_MODEL_ID"]

    return config


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

