import os
import pickle
from typing import Dict, List, Sequence, Tuple

SPEAKER_IDS = {"F": 0, "M": 1}


def utterance_id(audio_path: str) -> str:
    return os.path.splitext(os.path.basename(audio_path))[0]


def load_meta(data_root: str) -> Dict[str, dict]:
    path = os.path.join(data_root, "meta.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. This data was preprocessed before turn-order metadata existed; "
            "rerun scripts/preprocess.py."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def group_conversations(
    flat: Sequence[tuple], meta: Dict[str, dict]
) -> List[List[Tuple[float, int, str, str, int]]]:
    """Group (audio_path, text, label) samples into dialogues, turns in spoken order."""
    convs: Dict[str, list] = {}
    for audio_path, text, label in flat:
        uid = utterance_id(audio_path)
        if uid not in meta:
            raise KeyError(f"No metadata for {uid}; rerun scripts/preprocess.py")
        m = meta[uid]
        convs.setdefault(m["dialog"], []).append(
            (m["start"], SPEAKER_IDS[m["speaker"]], audio_path, text, int(label))
        )
    return [sorted(convs[d], key=lambda turn: turn[0]) for d in sorted(convs)]
