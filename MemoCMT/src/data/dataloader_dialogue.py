import os, pickle, re
from typing import List, Tuple, Dict
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import torchaudio

from configs.base import Config
from data.conversations import group_conversations, load_meta

def _text_preprocessing(text: str) -> str:
    text = re.sub("[\\(\\[].*?[\\)\\]]", "", str(text))
    text = re.sub(" +", " ", text).strip()
    try:
        text = " ".join(text.split())
    except:
        text = "NULL"
    if not text.strip():
        text = "NULL"
    return text

class ConversationDataset(Dataset):
    def __init__(self, cfg: Config, data_mode: str):
        super().__init__()
        self.cfg = cfg
        with open(os.path.join(cfg.data_root, data_mode), "rb") as f:
            flat = pickle.load(f)

        self.audio_max_length = cfg.audio_max_length
        self.text_max_length = cfg.text_max_length
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.pad_id = self.tokenizer.pad_token_id

        self.conversations = group_conversations(flat, load_meta(cfg.data_root))

    def __len__(self):
        return len(self.conversations)

    def _paudio(self, file_path: str) -> torch.Tensor:
        samples, sr = sf.read(file_path, dtype="int16")
        # Normalize to [-1, 1] if not already
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        elif samples.max() > 1.0 or samples.min() < -1.0:
            max_val = max(abs(samples.max()), abs(samples.min()))
            samples = samples / max_val
        
        if self.audio_max_length is not None and samples.shape[0] < self.audio_max_length:
            samples = np.pad(samples, (0, self.audio_max_length - samples.shape[0]), "constant")
        elif self.audio_max_length is not None:
            samples = samples[: self.audio_max_length]
        samples = torchaudio.functional.resample(samples, sr, 16000)
        tensor = torch.from_numpy(samples.astype(np.float32))
        if getattr(self, "cfg", None) is not None and getattr(self.cfg, "ablate_audio", False):
            tensor = torch.zeros_like(tensor)
        return tensor

    def _ptext(self, text: str) -> torch.Tensor:
        text = _text_preprocessing(text)
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        if self.text_max_length is not None and len(ids) < self.text_max_length:
            ids = np.pad(ids, (0, self.text_max_length - len(ids)), "constant", constant_values=self.pad_id)
        elif self.text_max_length is not None:
            ids = ids[: self.text_max_length]
        return torch.from_numpy(np.asarray(ids))

    def __getitem__(self, idx: int):
        seq = self.conversations[idx]
        T = len(seq)
        text_ids = []
        audio = []
        labels = []
        speakers = []
        for _, spk, apath, t, lab in seq:
            text_ids.append(self._ptext(t))
            audio.append(self._paudio(apath))
            labels.append(int(lab))
            speakers.append(int(spk))
        return {
            "text_ids": torch.stack(text_ids, dim=0),     # (T, L)
            "audio": torch.stack(audio, dim=0),           # (T, A)
            "labels": torch.tensor(labels, dtype=torch.long),    # (T,)
            "speakers": torch.tensor(speakers, dtype=torch.long),# (T,)
            "length": torch.tensor(T, dtype=torch.long),
            "pad_id": self.pad_id,
        }

def collate_conversations(batch: List[Dict]):
    B = len(batch)
    T_max = max(int(x["length"]) for x in batch)
    L = batch[0]["text_ids"].shape[1]
    A = batch[0]["audio"].shape[1]
    pad_id = int(batch[0]["pad_id"])

    text = torch.full((B, T_max, L), pad_id, dtype=torch.long)
    audio = torch.zeros(B, T_max, A, dtype=torch.float32)
    labels = torch.zeros(B, T_max, dtype=torch.long)
    speakers = torch.zeros(B, T_max, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    mask = torch.zeros(B, T_max, dtype=torch.bool)

    for b, item in enumerate(batch):
        T = int(item["length"])
        text[b, :T] = item["text_ids"]
        audio[b, :T] = item["audio"]
        labels[b, :T] = item["labels"]
        speakers[b, :T] = item["speakers"]
        lengths[b] = T
        mask[b, :T] = True

    return (text, audio, speakers, lengths, labels, mask)

def build_conversation_train_test_dataset(cfg: Config):
    train_data = ConversationDataset(cfg, "train.pkl")
    valid_set = cfg.data_valid if cfg.data_valid is not None else "test.pkl"
    test_data = ConversationDataset(cfg, valid_set)

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=getattr(cfg, "pin_memory", False),
        persistent_workers=getattr(cfg, "persistent_workers", False) and cfg.num_workers > 0,
        prefetch_factor=getattr(cfg, "prefetch_factor", 2) if cfg.num_workers > 0 else None,
        collate_fn=collate_conversations,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=getattr(cfg, "pin_memory", False),
        persistent_workers=getattr(cfg, "persistent_workers", False) and cfg.num_workers > 0,
        prefetch_factor=getattr(cfg, "prefetch_factor", 2) if cfg.num_workers > 0 else None,
        collate_fn=collate_conversations,
    )
    return (train_loader, test_loader)


def build_conversation_eval_loader(cfg: Config, split: str) -> DataLoader:
    return DataLoader(
        ConversationDataset(cfg, split),
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_conversations,
    )