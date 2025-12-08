import torch
import torch.nn as nn
import contextlib

from configs.base import Config
from torch.utils.checkpoint import checkpoint

from .modules import build_audio_encoder, build_text_encoder


class MemoCMT(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: str = "cpu",
    ):
        super(MemoCMT, self).__init__()
        # Text module
        self.text_encoder = build_text_encoder(cfg.text_encoder_type)
        self.text_encoder.to(device)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        if getattr(cfg, "use_gradient_checkpointing", False):
            try:
                self.text_encoder.gradient_checkpointing_enable()
                self.text_encoder.config.use_cache = False
            except Exception:
                pass
        if getattr(cfg, "text_unfreeze", False):
            try:
                base = getattr(self.text_encoder, "base_model", self.text_encoder)
                layers = base.encoder.layer
                num_layers = len(layers)
                unfreeze_layers = getattr(cfg, "text_unfreeze_layers", [-1, -2])
                for idx in unfreeze_layers:
                    i = num_layers + idx if idx < 0 else idx
                    if 0 <= i < num_layers:
                        for p in layers[i].parameters():
                            p.requires_grad = True
            except Exception:
                for p in self.text_encoder.parameters():
                    p.requires_grad = True

        # Audio module
        self.audio_encoder = build_audio_encoder(cfg)
        self.audio_encoder.to(device)
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        if getattr(cfg, "audio_unfreeze", False):
            n = int(getattr(cfg, "audio_unfreeze_last_n", 0))
            try:
                m = getattr(self.audio_encoder, "model", self.audio_encoder)
                if hasattr(m, "feature_extractor"):
                    for p in m.feature_extractor.parameters():
                        p.requires_grad = False
                layers = None
                if hasattr(m, "encoder") and hasattr(m.encoder, "transformer") and hasattr(m.encoder.transformer, "layers"):
                    layers = m.encoder.transformer.layers
                elif hasattr(m, "transformer") and hasattr(m.transformer, "layers"):
                    layers = m.transformer.layers
                if layers is not None and n > 0:
                    for layer in layers[-n:]:
                        for p in layer.parameters():
                            p.requires_grad = True
            except Exception:
                for p in self.audio_encoder.parameters():
                    p.requires_grad = True

        # Fusion module
        self.text_attention = nn.MultiheadAttention(
            embed_dim=cfg.text_encoder_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.text_linear = nn.Linear(cfg.text_encoder_dim, cfg.fusion_dim)
        self.text_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.audio_attention = nn.MultiheadAttention(
            embed_dim=cfg.audio_encoder_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.audio_linear = nn.Linear(cfg.audio_encoder_dim, cfg.fusion_dim)
        self.audio_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=cfg.fusion_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.fusion_linear = nn.Linear(cfg.fusion_dim, cfg.fusion_dim)
        self.fusion_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.dropout = nn.Dropout(cfg.dropout)

        self.linear_layer_output = cfg.linear_layer_output
        self.fusion_head_output_type = cfg.fusion_head_output_type

        # If using CLS pooling, fused feature is concatenation of text/audio (2 * fusion_dim)
        previous_dim = cfg.fusion_dim * 2 if self.fusion_head_output_type == "cls" else cfg.fusion_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)
        

    def forward(
        self,
        input_text: torch.Tensor,
        input_audio: torch.Tensor,
        output_attentions: bool = False,
    ):

        pad_id = getattr(self.text_encoder.config, "pad_token_id", 0)
        attention_mask = (input_text != pad_id).long()
        text_embeddings = self.text_encoder(input_text, attention_mask=attention_mask).last_hidden_state
        if len(input_audio.size()) != 2:
            batch_size, num_samples = input_audio.size(0), input_audio.size(1)
            audio_embeddings = self.audio_encoder(
                input_audio.view(-1, *input_audio.shape[2:])
            ).last_hidden_state
            audio_embeddings = audio_embeddings.mean(1)
            audio_embeddings = audio_embeddings.view(
                batch_size, num_samples, *audio_embeddings.shape[1:]
            )
        else:
            audio_embeddings = self.audio_encoder(input_audio)

        ## Fusion Module

        # Text cross attenttion text Q audio , K and V text
        text_attention, text_attn_output_weights = self.text_attention(
            audio_embeddings,
            text_embeddings,
            text_embeddings,
            average_attn_weights=False,
        )
        text_linear = self.text_linear(text_attention)
        text_norm = self.text_layer_norm(text_linear)
        text_norm = self.dropout(text_norm)

        # Audio cross attetntion Q text, K and V audio
        audio_attention, audio_attn_output_weights = self.audio_attention(
            text_embeddings,
            audio_embeddings,
            audio_embeddings,
            average_attn_weights=False,
        )
        audio_linear = self.audio_linear(audio_attention)
        audio_norm = self.audio_layer_norm(audio_linear)
        audio_norm = self.dropout(audio_norm)

        # Concatenate the text and audio embeddings
        fusion_norm = torch.cat((text_norm, audio_norm), 1)
        fusion_norm = self.dropout(fusion_norm)

        # Get classification output
        if self.fusion_head_output_type == "cls":
            # cls_token_final_fusion_norm = fusion_norm[:, 0, :]
            text_cls = text_norm[:, 0, :]  # Get BERT's CLS token features
            audio_aligned = audio_norm[:, 0, :]  # Get audio aligned to CLS
            cls_token_final_fusion_norm = torch.cat([text_cls, audio_aligned], dim=-1)
        elif self.fusion_head_output_type == "mean":
            cls_token_final_fusion_norm = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            cls_token_final_fusion_norm = fusion_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            cls_token_final_fusion_norm = fusion_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        # Classification head
        x = cls_token_final_fusion_norm
        x = self.dropout(x)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        out = self.classifer(x)

        if output_attentions:
            return [out, cls_token_final_fusion_norm], [
                text_attn_output_weights,
                audio_attn_output_weights,
            ]

        return out, cls_token_final_fusion_norm, text_norm, audio_norm

    def encode_audio(self, audio: torch.Tensor):
        return self.audio_encoder(audio)

    def encode_text(self, input_ids: torch.Tensor):
        pad_id = getattr(self.text_encoder.config, "pad_token_id", 0)
        attention_mask = (input_ids != pad_id).long()
        return self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state


class TextOnly(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: str = "cpu",
    ):
        super(TextOnly, self).__init__()
        # Text module
        self.text_encoder = build_text_encoder(cfg.text_encoder_type)
        self.text_encoder.to(device)
        
        # Freeze/Unfreeze the text module
        if not cfg.text_unfreeze:
            # Keep entire BERT frozen
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        else:
            # Default: freeze everything first
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            
            # Get number of layers in BERT
            num_layers = len(self.text_encoder.base_model.encoder.layer)
            
            # Convert negative indices to positive
            unfreeze_layers = getattr(cfg, 'text_unfreeze_layers', [-1, -2])  # Default: last 2 layers
            layers_to_unfreeze = [num_layers + idx if idx < 0 else idx for idx in unfreeze_layers]
            
            # Unfreeze specified layers
            for layer_idx in layers_to_unfreeze:
                if 0 <= layer_idx < num_layers:  # Ensure valid layer index
                    for param in self.text_encoder.base_model.encoder.layer[layer_idx].parameters():
                        param.requires_grad = True
            
            # Log which layers are unfrozen
            unfrozen_layers = [i for i in range(num_layers)
                              if any(p.requires_grad for p in self.text_encoder.base_model.encoder.layer[i].parameters())]
            print(f'Unfrozen BERT layers: {unfrozen_layers} out of {num_layers} total layers')

        self.dropout = nn.Dropout(cfg.dropout)

        self.linear_layer_output = cfg.linear_layer_output

        previous_dim = cfg.text_encoder_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

        self.fusion_head_output_type = cfg.fusion_head_output_type

    def forward(
        self,
        input_text: torch.Tensor,
        input_audio: torch.Tensor,
        output_attentions: bool = False,
    ):

        text_embeddings = self.text_encoder(input_text).last_hidden_state
        fusion_norm = self.dropout(text_embeddings)

        # Get classification output
        if self.fusion_head_output_type == "cls":
            cls_token_final_fusion_norm = fusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            cls_token_final_fusion_norm = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            cls_token_final_fusion_norm = fusion_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            cls_token_final_fusion_norm = fusion_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        # Classification head
        x = cls_token_final_fusion_norm
        x = self.dropout(x)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        out = self.classifer(x)

        return out, cls_token_final_fusion_norm


class AudioOnly(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: str = "cpu",
    ):
        super(AudioOnly, self).__init__()

        # Audio module
        self.audio_encoder = build_audio_encoder(cfg)
        self.audio_encoder.to(device)

        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = cfg.audio_unfreeze

        self.linear_layer_output = cfg.linear_layer_output

        self.dropout = nn.Dropout(cfg.dropout)

        previous_dim = cfg.audio_encoder_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

        self.fusion_head_output_type = cfg.fusion_head_output_type

    def forward(
        self,
        input_text: torch.Tensor,
        input_audio: torch.Tensor,
        output_attentions: bool = False,
    ):

        if len(input_audio.size()) != 2:
            batch_size, num_samples = input_audio.size(0), input_audio.size(1)
            audio_embeddings = self.audio_encoder(
                input_audio.view(-1, *input_audio.shape[2:])
            ).last_hidden_state
            audio_embeddings = audio_embeddings.mean(1)
            audio_embeddings = audio_embeddings.view(
                batch_size, num_samples, *audio_embeddings.shape[1:]
            )
        else:
            audio_embeddings = self.audio_encoder(input_audio)

        fusion_norm = self.dropout(audio_embeddings)

        # Get classification output
        if self.fusion_head_output_type == "cls":
            cls_token_final_fusion_norm = fusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            cls_token_final_fusion_norm = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            cls_token_final_fusion_norm = fusion_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            cls_token_final_fusion_norm = fusion_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        # Classification head
        x = cls_token_final_fusion_norm
        x = self.dropout(x)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        out = self.classifer(x)

        return out, cls_token_final_fusion_norm

class DialogueRNNCell(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, num_speakers: int = 2, window: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_speakers = num_speakers
        self.window = window
        self.g_gru = nn.GRUCell(feat_dim + hidden_dim, hidden_dim)
        self.p_gru = nn.GRUCell(feat_dim, hidden_dim)
        self.e_gru = nn.GRUCell(feat_dim + hidden_dim, hidden_dim)

    def _attend(self, hist, q, device):
        if len(hist) == 0:
            return torch.zeros(self.hidden_dim, device=device)
        H = torch.stack(hist[-self.window:])  # (w, H)
        w = torch.softmax(H @ q, dim=0)       # (w,)
        return (w.unsqueeze(1) * H).sum(0)

    def forward(self, u_seq: torch.Tensor, speakers: torch.Tensor, lengths: torch.Tensor):
        B, T, D = u_seq.shape
        H = self.hidden_dim
        e_seq = torch.zeros(B, T, H, device=u_seq.device)
        for b in range(B):
            g = torch.zeros(H, device=u_seq.device)
            e = torch.zeros(H, device=u_seq.device)
            party = [torch.zeros(H, device=u_seq.device) for _ in range(self.num_speakers)]
            g_hist = []
            T_b = int(lengths[b].item())
            for t in range(T_b):
                u_t = u_seq[b, t]
                s_t = int(speakers[b, t].item()) if speakers is not None else 0
                c_t = self._attend(g_hist, g, u_seq.device)
                g = self.g_gru(torch.cat([u_t, c_t], -1), g)
                party[s_t] = self.p_gru(u_t, party[s_t])
                e = self.e_gru(torch.cat([u_t, c_t], -1), e)
                g_hist.append(g)
                e_seq[b, t] = e
        return e_seq

class MemoCMTDialogueRNN(nn.Module):
    def __init__(self, cfg: Config, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.memo = MemoCMT(cfg, device=device)
        # Selectively freeze only encoders if requested, leave fusion/cross-attn trainable
        freeze = getattr(cfg, "freeze_feature_extractor", True)
        # for p in self.memo.parameters():
        #     p.requires_grad = not freeze
        if freeze:
            # self.memo.eval()
            for p in self.memo.text_encoder.parameters():
                p.requires_grad = False
            for p in self.memo.audio_encoder.parameters():
                p.requires_grad = False
            # Keep encoders in eval for stable batchnorm/layernorm stats
            self.memo.text_encoder.eval()
            self.memo.audio_encoder.eval()

        # Feature size produced by MemoCMT fusion head
        feat_dim_ctx = cfg.fusion_dim * 2 if getattr(cfg, "fusion_head_output_type", "cls") == "cls" else cfg.fusion_dim
        self.ctx = DialogueRNNCell(
            # feat_dim=cfg.fusion_dim,
            feat_dim=feat_dim_ctx,
            hidden_dim=cfg.dialogue_hidden_size,
            num_speakers=getattr(cfg, "num_speakers", 2),
            window=getattr(cfg, "context_window", 8),
        )

        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.dialogue_hidden_size, cfg.num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_text: torch.Tensor,     # (B, T, L)
        input_audio: torch.Tensor,    # (B, T, A)
        speakers: torch.Tensor,       # (B, T)
        lengths: torch.Tensor,        # (B,)
    ):
        B, T, L = input_text.shape
        A = input_audio.shape[-1]
        text_flat = input_text.view(B * T, L)
        audio_flat = input_audio.view(B * T, A)

        # with torch.no_grad() if all(not p.requires_grad for p in self.memo.parameters()) else torch.enable_grad():
        #     _, fused_flat, _, _ = self.memo(text_flat, audio_flat)

        # Microbatch the feature extractor to limit peak memory; restrict autocast to feature extraction
        chunk_size = int(getattr(self.cfg, "memo_chunk_size", 16))
        fused_chunks = []

        # Determine if any encoder params require grad to choose grad/no_grad context
        enc_params = list(self.memo.text_encoder.parameters()) + list(self.memo.audio_encoder.parameters())
        enc_trainable = any(p.requires_grad for p in enc_params)

        # Use autocast only for the heavy feature extractor part
        use_amp = getattr(self.cfg, "use_amp", False) and (torch.cuda.is_available())
        amp_ctx = torch.amp.autocast('cuda', dtype=torch.float16) if use_amp else contextlib.nullcontext()

        grad_ctx = torch.enable_grad() if enc_trainable else torch.no_grad()
        with grad_ctx:
            with amp_ctx:
                # define small wrapper for checkpointing fused feature
                def _memo_fused(tt, aa):
                    return self.memo(tt, aa)[1]
                for i in range(0, B * T, chunk_size):
                    j = min(i + chunk_size, B * T)
                    if enc_trainable:
                        fused_chunk = checkpoint(_memo_fused, text_flat[i:j], audio_flat[i:j], use_reentrant=False)
                    else:
                        _, fused_chunk, _, _ = self.memo(text_flat[i:j], audio_flat[i:j])
                    fused_chunks.append(fused_chunk)

        fused_flat = torch.cat(fused_chunks, dim=0)
        fused = fused_flat.view(B, T, -1)

        # Run context RNN and classifier in FP32 for stability
        e_seq = self.ctx(fused, speakers, lengths)          # (B, T, H)
        logits = self.classifier(self.dropout(e_seq))       # (B, T, C)
        return logits, e_seq

__all__ = ['MemoCMT', 'TextOnly', 'AudioOnly', 'MemoCMTDialogueRNN']