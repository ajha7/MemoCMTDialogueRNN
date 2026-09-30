from types import SimpleNamespace

import pytest

from utils.configs import apply_overrides, check_selection_config, parse_overrides


def test_parse_overrides_types():
    assert parse_overrides(["seed=1", "context_window=None", "ablate_audio=True", "name=w6_seed1", "lr=5e-6"]) == {
        "seed": 1, "context_window": None, "ablate_audio": True, "name": "w6_seed1", "lr": 5e-6,
    }


def test_unknown_override_key_raises_instead_of_silently_training_the_default():
    cfg = SimpleNamespace(context_window=6)
    with pytest.raises(KeyError, match="contex_window"):
        apply_overrides(cfg, {"contex_window": 1})
    assert cfg.context_window == 6


def test_known_override_is_applied():
    cfg = SimpleNamespace(context_window=6)
    apply_overrides(cfg, {"context_window": None})
    assert cfg.context_window is None


def _selection_cfg(**kw):
    return SimpleNamespace(**{"save_best_val": True, "best_metric": "ua", "skip_first_epoch_eval": True,
                              "num_epochs": 25, **kw})


def test_selection_config_accepts_the_defaults():
    check_selection_config(_selection_cfg())


@pytest.mark.parametrize("bad", [
    {"save_best_val": False},
    {"best_metric": "UA"},
    {"num_epochs": 1},  # epoch 1 validation is skipped, so no best checkpoint would exist
])
def test_selection_config_fails_before_training(bad):
    with pytest.raises(ValueError):
        check_selection_config(_selection_cfg(**bad))
