import torch.nn as nn
import logging
import os
from typing import Dict

import torch
from torch import Tensor
from configs.base import Config
from models.networks import MemoCMT
from utils.torch.trainer import TorchTrainer

BERT_CLS, BERT_SEP, BERT_PAD = 101, 102, 0  # bert-base-uncased


def _blank_text(ids: Tensor) -> Tensor:
    """Replace every transcript with just [CLS][SEP] so BERT gets no lexical content."""
    blank = torch.full_like(ids, BERT_PAD)
    blank[..., 0] = BERT_CLS
    blank[..., 1] = BERT_SEP
    return blank


class Trainer(TorchTrainer):
    def __init__(
        self,
        cfg: Config,
        network: MemoCMT,
        criterion: torch.nn.CrossEntropyLoss = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.network = network
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        
        # AMP scaler
        # self.scaler = torch.cuda.amp.GradScaler(enabled=getattr(self.cfg, "use_amp", False))
        # In trainer.py __init__:
        self.scaler = torch.amp.GradScaler(
            'cuda',
            enabled=getattr(self.cfg, "use_amp", False),
            init_scale=2**16,        # Start smaller
            growth_factor=1.5,       # Grow more slowly
            backoff_factor=0.5,      # Back off more aggressively
            growth_interval=2000     # Wait longer between scales
        )
        
    # def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
    #     self.network.train()
    #     self.optimizer.zero_grad(set_to_none=True)
        
    #     # Prepare batch
    #     input_text, input_audio, label = batch
        
    #     # Move inputs to device (non_blocking if pin_memory)
    #     input_audio = input_audio.to(self.device, non_blocking=True)
    #     label = label.to(self.device, non_blocking=True)
    #     input_text = input_text.to(self.device, non_blocking=True)
        
    #     use_amp = getattr(self.cfg, "use_amp", False) and self.device.type == "cuda"
    #     if use_amp:
    #         with torch.autocast(device_type="cuda", dtype=torch.float16):
    #             output = self.network(input_text, input_audio)
    #             loss = self.criterion(output, label)
    #         self.scaler.scale(loss).backward()
    #         self.scaler.step(self.optimizer)
    #         self.scaler.update()
    #     else:
    #         output = self.network(input_text, input_audio)
    #         loss = self.criterion(output, label)
    #         loss.backward()
    #         self.optimizer.step()

    #     # Calculate accuracy
    #     _, preds = torch.max(output[0], 1)
    #     accuracy = torch.mean((preds == label).float())
    #     return {
    #         "loss": loss.detach().cpu().item(),
    #         "acc": accuracy.detach().cpu().item(),
    #     }

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        # Prepare batch
        input_text, input_audio, label = batch
        
        # Move inputs to device
        input_audio = input_audio.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)
        input_text = input_text.to(self.device, non_blocking=True)
        if getattr(self.cfg, "ablate_audio", False):
            input_audio = torch.zeros_like(input_audio)
        
        use_amp = getattr(self.cfg, "use_amp", False) and self.device.type == "cuda"
        try:
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    output = self.network(input_text, input_audio)
                    loss = self.criterion(output[0], label)

                # Check for NaN before scaling
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Loss is {loss.item()}")
                    
                self.scaler.scale(loss).backward()
                
                # Add gradient clipping
                if hasattr(self.cfg, "max_grad_norm"):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), 
                        self.cfg.max_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.network(input_text, input_audio)
                loss = self.criterion(output[0], label)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Loss is {loss.item()}")
                    
                loss.backward()
                
                # Add gradient clipping
                if hasattr(self.cfg, "max_grad_norm"):
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), 
                        self.cfg.max_grad_norm
                    )
                
                self.optimizer.step()

        except RuntimeError as e:
            logging.warning(f"Caught error in training step: {str(e)}")
            return {
                "loss": float('nan'),
                "acc": 0.0,
            }

        # Calculate accuracy
        _, preds = torch.max(output[0], 1)
        accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }
    
    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        
        # Prepare batch
        input_text, input_audio, label = batch
        
        # Move inputs to device
        input_audio = input_audio.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)
        input_text = input_text.to(self.device, non_blocking=True)
        if getattr(self.cfg, "ablate_audio", False):
            input_audio = torch.zeros_like(input_audio)
        
        with torch.no_grad():
            use_amp = getattr(self.cfg, "use_amp", False) and self.device.type == "cuda"
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    output = self.network(input_text, input_audio)
                    loss = self.criterion(output[0], label)
            else:
                output = self.network(input_text, input_audio)
                loss = self.criterion(output[0], label)
            # Calculate accuracy
            _, preds = torch.max(output[0], 1)
            accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
            "preds": preds.cpu().numpy(),
            "labels": label.cpu().numpy(),
        }

class DialogueTrainer(TorchTrainer):
    def __init__(self, cfg: Config, network: nn.Module, criterion=None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.network = network
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.scaler = torch.amp.GradScaler(
            'cuda',
            enabled=getattr(self.cfg, "use_amp", False),
            init_scale=2**16,
            growth_factor=1.5,
            backoff_factor=0.5,
            growth_interval=2000,
        )

    def _step(self, batch, train: bool):
        input_text, input_audio, speakers, lengths, labels, mask = batch
        input_text = input_text.to(self.device, non_blocking=True)
        input_audio = input_audio.to(self.device, non_blocking=True)
        speakers = speakers.to(self.device, non_blocking=True)
        lengths = lengths.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        if getattr(self.cfg, "ablate_audio", False):
            input_audio = torch.zeros_like(input_audio)
        if getattr(self.cfg, "ablate_text", False):
            input_text = _blank_text(input_text)

        def compute():
            logits, _ = self.network(input_text, input_audio, speakers, lengths)  # (B,T,C)
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            labels_flat = labels.view(B * T)
            mask_flat = mask.view(B * T)
            logits_sel = logits_flat[mask_flat]
            labels_sel = labels_flat[mask_flat]
            loss = self.criterion(logits_sel, labels_sel)
            with torch.no_grad():
                preds = torch.argmax(logits_sel, dim=-1)
            return loss, preds, labels_sel

        if train:
            self.network.train()
            self.optimizer.zero_grad(set_to_none=True)
            use_amp = getattr(self.cfg, "use_amp", False) and self.device.type == "cuda"
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    loss, preds, labels_sel = compute()
                    # Check for NaN
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Loss is {loss.item()}")
                self.scaler.scale(loss).backward()

                # if hasattr(self.cfg, "max_grad_norm"):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, preds, labels_sel = compute()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Loss is {loss.item()}")
                loss.backward()
                if hasattr(self.cfg, "max_grad_norm"):
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
        else:
            self.network.eval()
            use_amp = getattr(self.cfg, "use_amp", False) and self.device.type == "cuda"
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        loss, preds, labels_sel = compute()
                else:
                    loss, preds, labels_sel = compute()

        out = {"loss": float(loss.detach().cpu()), "acc": float((preds == labels_sel).float().mean().cpu())}
        if not train:
            out["preds"] = preds.detach().cpu().numpy()
            out["labels"] = labels_sel.detach().cpu().numpy()
        return out

    def train_step(self, batch: Dict[str, Tensor]):
        return self._step(batch, train=True)

    def test_step(self, batch: Dict[str, Tensor]):
        return self._step(batch, train=False)