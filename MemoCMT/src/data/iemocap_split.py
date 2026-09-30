import random
from typing import Dict, List, Sequence, Tuple

from data.conversations import utterance_id


def split_by_session(
    samples: Sequence[tuple],
    meta: Dict[str, dict],
    test_session: int = 5,
    val_frac: float = 0.1,
    seed: int = 0,
) -> Tuple[List[tuple], List[tuple], List[tuple]]:
    """Test = every utterance from test_session. Val = val_frac of the remaining
    dialogues, so no dialogue (or speaker pair) straddles train and test."""
    session = lambda s: meta[utterance_id(s[0])]["session"]
    dialog = lambda s: meta[utterance_id(s[0])]["dialog"]

    test = [s for s in samples if session(s) == test_session]
    rest = [s for s in samples if session(s) != test_session]

    dialogs = sorted({dialog(s) for s in rest})
    random.Random(seed).shuffle(dialogs)
    val_dialogs = set(dialogs[: max(1, round(len(dialogs) * val_frac))])

    train = [s for s in rest if dialog(s) not in val_dialogs]
    val = [s for s in rest if dialog(s) in val_dialogs]
    return train, val, test
