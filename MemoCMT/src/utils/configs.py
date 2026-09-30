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
