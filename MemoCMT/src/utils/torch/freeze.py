from typing import Dict

from torch import nn

ENCODERS = ("text_encoder", "audio_encoder")


def _encoders(network: nn.Module):
    base = getattr(network, "memo", network)
    return [getattr(base, name) for name in ENCODERS if hasattr(base, name)]


def freeze_encoders(network: nn.Module) -> Dict[nn.Parameter, bool]:
    """Freeze the text/audio encoders (MemoCMT or MemoCMTDialogueRNN) for a fast first epoch.

    Returns each encoder parameter's previous requires_grad so restore_encoders can put back
    exactly the set the optimizer was built with.
    """
    snapshot = {}
    for encoder in _encoders(network):
        for p in encoder.parameters():
            snapshot[p] = p.requires_grad
            p.requires_grad = False
        encoder.eval()
    return snapshot


def restore_encoders(network: nn.Module, snapshot: Dict[nn.Parameter, bool]) -> None:
    for p, requires_grad in snapshot.items():
        p.requires_grad = requires_grad
    for encoder in _encoders(network):
        encoder.train(any(p.requires_grad for p in encoder.parameters()))
