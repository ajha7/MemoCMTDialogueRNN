import numpy as np
import pytest

from utils.metrics import aggregate_eval_outputs, compute_metrics

Y_TRUE = [0, 0, 0, 0, 1, 1, 2, 3]
Y_PRED = [0, 0, 0, 0, 0, 1, 2, 2]


def test_hand_computed_values():
    m = compute_metrics(Y_TRUE, Y_PRED, num_classes=4)
    assert m["wa"] == pytest.approx(6 / 8)
    assert m["ua"] == pytest.approx((1 + 0.5 + 1 + 0) / 4)
    assert m["macro_f1"] == pytest.approx(5 / 9)
    assert m["per_class_recall"] == pytest.approx([1.0, 0.5, 1.0, 0.0])
    assert m["confusion_matrix"][1] == [1, 1, 0, 0]


def test_missing_class_still_gives_full_shapes():
    m = compute_metrics([0, 0, 1], [0, 0, 0], num_classes=4)
    assert len(m["per_class_recall"]) == 4
    assert np.array(m["confusion_matrix"]).shape == (4, 4)


def test_pools_utterances_instead_of_averaging_conversations():
    # 1-turn conversation all right, 3-turn conversation all wrong.
    # Mean of per-conversation accuracy = 0.5; true utterance accuracy = 0.25.
    outputs = [
        {"loss": 1.0, "preds": np.array([0]), "labels": np.array([0])},
        {"loss": 3.0, "preds": np.array([1, 1, 1]), "labels": np.array([0, 0, 0])},
    ]
    m = aggregate_eval_outputs(outputs, num_classes=4)
    assert m["wa"] == pytest.approx(0.25)
    assert m["loss"] == pytest.approx(2.0)
