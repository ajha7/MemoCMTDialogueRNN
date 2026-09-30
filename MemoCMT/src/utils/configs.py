import ast
import importlib
import sys
from typing import List

from configs.base import BaseConfig


def get_options(
    path: str,
) -> BaseConfig:
    """Get arguments for training and evaluate
    Returns:
        cfg: ArgumentParser
    """
    # Import config from path
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    options = config.Config()
    return options


def parse_overrides(pairs: List[str]) -> dict:
    out = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key] = value
    return out


def apply_overrides(cfg, overrides: dict) -> None:
    """Set --set values on cfg. Unknown keys are typos, so fail instead of training the default."""
    unknown = [key for key in overrides if not hasattr(cfg, key)]
    if unknown:
        raise KeyError(f"Unknown config keys in --set: {unknown}")
    for key, value in overrides.items():
        setattr(cfg, key, value)


SELECTION_METRICS = ("ua", "wa", "macro_f1", "weighted_f1", "loss")


def check_selection_config(cfg) -> None:
    """Fail before training if the run could never produce a best-val checkpoint to test."""
    if not getattr(cfg, "save_best_val", False):
        raise ValueError("save_best_val must be True: the test set is scored from the best-val checkpoint.")
    if getattr(cfg, "best_metric", "ua") not in SELECTION_METRICS:
        raise ValueError(f"best_metric={cfg.best_metric!r} is not one of {SELECTION_METRICS}.")
    if getattr(cfg, "skip_first_epoch_eval", True) and cfg.num_epochs < 2:
        raise ValueError("num_epochs must be >= 2 when skip_first_epoch_eval is on; epoch 1 is never validated.")
