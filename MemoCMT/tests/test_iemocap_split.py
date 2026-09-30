from data.conversations import utterance_id
from data.iemocap_split import split_by_session


def _fake(num_sessions=5, dialogs_per_session=10, turns=4):
    samples, meta = [], {}
    for s in range(1, num_sessions + 1):
        for d in range(dialogs_per_session):
            dialog = f"Ses0{s}F_impro{d:02d}"
            for t in range(turns):
                uid = f"{dialog}_{'FM'[t % 2]}{t:03d}"
                samples.append((f"/data/{uid}.wav", f"text {uid}", t % 4))
                meta[uid] = {"session": s, "dialog": dialog, "speaker": "FM"[t % 2], "start": float(t)}
    return samples, meta


def test_utterance_id_strips_dir_and_extension():
    assert utterance_id("/a/b/Ses01F_impro01_F000.wav") == "Ses01F_impro01_F000"


def test_test_split_is_exactly_the_held_out_session():
    samples, meta = _fake()
    _, _, test = split_by_session(samples, meta, test_session=5)
    assert test and all(meta[utterance_id(s[0])]["session"] == 5 for s in test)
    assert len(test) == 10 * 4


def test_no_dialogue_is_split_across_train_val_test():
    samples, meta = _fake()
    train, val, test = split_by_session(samples, meta)
    dialogs = lambda split: {meta[utterance_id(s[0])]["dialog"] for s in split}
    assert not dialogs(train) & dialogs(val)
    assert not dialogs(train) & dialogs(test)
    assert not dialogs(val) & dialogs(test)


def test_val_is_about_val_frac_of_non_test_dialogues():
    samples, meta = _fake()
    _, val, _ = split_by_session(samples, meta, val_frac=0.1)
    assert len({meta[utterance_id(s[0])]["dialog"] for s in val}) == 4  # 10% of 40


def test_split_is_a_partition_and_deterministic():
    samples, meta = _fake()
    a = split_by_session(samples, meta, seed=0)
    b = split_by_session(samples, meta, seed=0)
    assert a == b
    assert sorted(a[0] + a[1] + a[2]) == sorted(samples)
