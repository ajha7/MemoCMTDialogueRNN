import pytest

from data.conversations import group_conversations, load_meta

D = "Ses01F_impro01"


def _meta(**starts):
    return {f"{D}_{k}": {"session": 1, "dialog": D, "speaker": k[0], "start": v} for k, v in starts.items()}


def test_orders_by_start_time_not_per_speaker_counter():
    # Filename counters are per speaker: F000, F001, M000. Real order is F000, F001, M000.
    # Sorting by the counter would give F000, M000, F001.
    flat = [(f"/x/{D}_M000.wav", "m0", 3), (f"/x/{D}_F001.wav", "f1", 1), (f"/x/{D}_F000.wav", "f0", 0)]
    [conv] = group_conversations(flat, _meta(F000=1.0, F001=2.0, M000=3.0))
    assert [turn[3] for turn in conv] == ["f0", "f1", "m0"]
    assert [turn[1] for turn in conv] == [0, 0, 1]
    assert [turn[4] for turn in conv] == [0, 1, 3]


def test_separate_dialogues_are_separate_conversations():
    meta = _meta(F000=0.0)
    meta["Ses01F_impro02_M000"] = {"session": 1, "dialog": "Ses01F_impro02", "speaker": "M", "start": 0.0}
    flat = [(f"/x/{D}_F000.wav", "a", 0), ("/x/Ses01F_impro02_M000.wav", "b", 1)]
    assert len(group_conversations(flat, meta)) == 2


def test_missing_metadata_raises():
    with pytest.raises(KeyError, match="preprocess.py"):
        group_conversations([(f"/x/{D}_F000.wav", "a", 0)], {})


def test_load_meta_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess.py"):
        load_meta(str(tmp_path))
