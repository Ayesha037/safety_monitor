from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "settings.yaml"

class Config:
    def __init__(self, data: dict):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> dict:
        return self._data

def load_config(path=_DEFAULT_CONFIG_PATH) -> Config:
    env_path = os.environ.get("SAFETY_CONFIG_PATH")
    if env_path:
        path = Path(env_path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)

cfg = load_config()