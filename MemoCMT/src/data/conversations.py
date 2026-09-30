import os


def utterance_id(audio_path: str) -> str:
    return os.path.splitext(os.path.basename(audio_path))[0]
