import torch.nn as nn

from utils.torch.freeze import freeze_encoders, restore_encoders


def _net():
    # Like MemoCMT with text_unfreeze: only BERT's last layer trains, HuBERT is frozen.
    net = nn.Module()
    net.memo = nn.Module()
    net.memo.text_encoder = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    net.memo.audio_encoder = nn.Linear(2, 2)
    net.ctx = nn.Linear(2, 2)
    for p in net.memo.parameters():
        p.requires_grad = False
    for p in net.memo.text_encoder[1].parameters():
        p.requires_grad = True
    return net


def test_freeze_stops_all_encoder_gradients():
    net = _net()
    freeze_encoders(net)
    assert not any(p.requires_grad for p in net.memo.parameters())
    assert all(p.requires_grad for p in net.ctx.parameters())


def test_restore_brings_back_exactly_the_trainable_set():
    # The optimizer only holds the parameters that were trainable when it was built.
    # Unfreezing more than that leaves gradients no optimizer ever zeroes or steps.
    net = _net()
    before = [p.requires_grad for p in net.parameters()]
    restore_encoders(net, freeze_encoders(net))
    assert [p.requires_grad for p in net.parameters()] == before
