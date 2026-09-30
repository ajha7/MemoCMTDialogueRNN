from typing import Optional

import torch
import torch.nn as nn


class DialogueRNNCell(nn.Module):
    """DialogueRNN (Majumder et al., 2019) over per-utterance features u_t.

    For turn t spoken by s:
        g_t     = GRU_G(u_t ⊕ q_{s,t-1}, g_{t-1})
        c_t     = Σ_{i<t} α_i g_i,   α = softmax_i(g_i · W_α u_t)
        q_{s,t} = GRU_P(u_t ⊕ c_t, q_{s,t-1})     (the listener's state is unchanged)
        e_t     = GRU_E(q_{s,t}, e_{t-1})

    window=w predicts turn t from a fresh pass over turns max(0, t-w+1)..t, so w=1 uses
    no conversational context. window=None makes one causal pass over the whole dialogue.
    """

    def __init__(self, feat_dim: int, hidden_dim: int, num_speakers: int = 2, window: Optional[int] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_speakers = num_speakers
        self.window = window
        self.g_gru = nn.GRUCell(feat_dim + hidden_dim, hidden_dim)
        self.p_gru = nn.GRUCell(feat_dim + hidden_dim, hidden_dim)
        self.e_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.attn = nn.Linear(feat_dim, hidden_dim, bias=False)

    def _run(self, u: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """u: (T, D), spk: (T,) -> emotion states (T, H), starting from zero states."""
        zeros = u.new_zeros(self.hidden_dim)
        g, e = zeros, zeros
        party = [zeros for _ in range(self.num_speakers)]
        g_hist, out = [], []
        for t in range(u.shape[0]):
            s = int(spk[t])
            if g_hist:
                G = torch.stack(g_hist)
                alpha = torch.softmax(G @ self.attn(u[t]), dim=0)
                c = (alpha.unsqueeze(1) * G).sum(0)
            else:
                c = zeros
            g = self.g_gru(torch.cat([u[t], party[s]], -1), g)
            party[s] = self.p_gru(torch.cat([u[t], c], -1), party[s])
            e = self.e_gru(party[s], e)
            g_hist.append(g)
            out.append(e)
        return torch.stack(out)

    def forward(self, u_seq: torch.Tensor, speakers: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = u_seq.shape
        e_seq = u_seq.new_zeros(B, T, self.hidden_dim)
        for b in range(B):
            T_b = int(lengths[b])
            u, spk = u_seq[b, :T_b], speakers[b, :T_b]
            if self.window is None:
                e_seq[b, :T_b] = self._run(u, spk)
            else:
                for t in range(T_b):
                    lo = max(0, t - self.window + 1)
                    e_seq[b, t] = self._run(u[lo : t + 1], spk[lo : t + 1])[-1]
        return e_seq


class EmotionClassifier(nn.Module):
    """DialogueRNN's output layer: l_t = ReLU(W_l e_t + b_l), logits_t = W_smax l_t + b_smax.

    Dropout is applied to e_t and to l_t, as in the original implementation. The softmax is
    left to the cross-entropy loss.
    """

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.smax_fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.linear(self.dropout(e)))
        return self.smax_fc(self.dropout(hidden))
