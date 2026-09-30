from types import SimpleNamespace

import torch

from models.dialogue_rnn import DialogueRNNCell, EmotionClassifier
from models.optims import _build_param_groups

D, H = 8, 6
ALT = torch.tensor([0, 1, 0, 1, 0, 1])


def make(window):
    torch.manual_seed(0)
    return DialogueRNNCell(D, H, num_speakers=2, window=window).eval()


def run(cell, u, spk=ALT):
    return cell(u.unsqueeze(0), spk.unsqueeze(0), torch.tensor([u.shape[0]]))[0]


def utts():
    torch.manual_seed(1)
    return torch.randn(6, D)


def test_shape_and_padding():
    cell = make(None)
    u = torch.randn(2, 5, D)
    spk = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 1, 0, 0]])
    out = cell(u, spk, torch.tensor([5, 3]))
    assert out.shape == (2, 5, H)
    assert torch.all(out[1, 3:] == 0)


def test_padding_does_not_change_real_turns():
    cell = make(None)
    u = utts()
    alone = run(cell, u[:3], ALT[:3])
    batched = cell(torch.stack([u, u]), torch.stack([ALT, ALT]), torch.tensor([6, 3]))[1, :3]
    assert torch.allclose(alone, batched, atol=1e-6)


def test_window_1_ignores_previous_turns():
    cell, u = make(1), utts()
    changed = u.clone()
    changed[:5] += 10.0
    assert torch.allclose(run(cell, changed)[5], run(cell, u)[5], atol=1e-6)


def test_window_3_sees_exactly_the_last_3_turns():
    cell, u = make(3), utts()
    base = run(cell, u)[5]
    outside = u.clone()
    outside[2] += 10.0  # turn 5 sees turns 3, 4, 5
    assert torch.allclose(run(cell, outside)[5], base, atol=1e-6)
    inside = u.clone()
    inside[3] += 10.0
    assert not torch.allclose(run(cell, inside)[5], base, atol=1e-4)


def test_full_context_carries_the_first_turn():
    cell, u = make(None), utts()
    changed = u.clone()
    changed[0] += 10.0
    assert not torch.allclose(run(cell, changed)[5], run(cell, u)[5], atol=1e-4)


def test_window_longer_than_conversation_equals_full():
    u = utts()
    assert torch.allclose(run(make(50), u), run(make(None), u), atol=1e-6)


def test_speaker_identity_matters():
    cell, u = make(None), utts()
    same_speaker = torch.zeros(6, dtype=torch.long)
    assert not torch.allclose(run(cell, u, ALT)[5], run(cell, u, same_speaker)[5], atol=1e-4)


def test_classifier_is_relu_then_linear():
    torch.manual_seed(0)
    head = EmotionClassifier(H, 4, dropout=0.5).eval()
    e = torch.randn(2, 3, H)
    expected = head.smax_fc(torch.relu(head.linear(e)))
    assert head(e).shape == (2, 3, 4)
    assert torch.allclose(head(e), expected)


def test_classifier_trains_at_the_dialogue_learning_rate():
    net = torch.nn.Module()
    net.ctx = DialogueRNNCell(D, H)
    net.classifier = EmotionClassifier(H, 4)
    cfg = SimpleNamespace(learning_rate=5e-6, dialogue_learning_rate=1e-4)
    head_ids = {id(p) for p in net.classifier.parameters()}
    for group in _build_param_groups(cfg, net):
        if head_ids & {id(p) for p in group["params"]}:
            assert group["lr"] == 1e-4
            assert head_ids <= {id(p) for p in group["params"]}
