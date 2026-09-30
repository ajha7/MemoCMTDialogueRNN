from utils.configs import parse_overrides


def test_parse_overrides_types():
    assert parse_overrides(["seed=1", "context_window=None", "ablate_audio=True", "name=w6_seed1", "lr=5e-6"]) == {
        "seed": 1, "context_window": None, "ablate_audio": True, "name": "w6_seed1", "lr": 5e-6,
    }
