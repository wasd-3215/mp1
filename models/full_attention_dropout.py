"""Combined Pre-RMSNorm/RoPE/GQA/SwiGLU model with configurable dropout.

Dropout is applied to the token embeddings (optionally) and to each attention
and feed-forward residual branch. Evaluation disables dropout via ``eval()``.
"""
import torch
import math
from torch import nn
from torch.nn import functional as F

from .transformer_components import CombinedGPT, GQARoPEAttention, RMSNorm


class ConfigurableSwiGLU(nn.Module):
    """SwiGLU FFN with a config-selectable hidden-width multiplier."""

    def __init__(self, width, hidden_multiplier=8.0 / 3.0, multiple=8):
        super().__init__()
        if hidden_multiplier <= 0:
            raise ValueError("ffn_hidden_multiplier must be positive")
        hidden = math.ceil((width * hidden_multiplier) / multiple) * multiple
        self.gate_proj = nn.Linear(width, hidden, bias=False)
        self.value_proj = nn.Linear(width, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.value_proj(x))


class DropoutTransformerBlock(nn.Module):
    def __init__(self, width, query_heads, kv_heads,
                 resid_dropout=0.0, ffn_hidden_multiplier=8.0 / 3.0):
        super().__init__()
        self.attn_norm = RMSNorm(width)
        self.attention = GQARoPEAttention(width, query_heads, kv_heads)
        self.attn_dropout = nn.Dropout(resid_dropout)
        self.ffn_norm = RMSNorm(width)
        self.ffn = ConfigurableSwiGLU(width, ffn_hidden_multiplier)
        self.ffn_dropout = nn.Dropout(resid_dropout)

    def forward(self, x):
        x = x + self.attn_dropout(self.attention(self.attn_norm(x)))
        return x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))


class CombinedGPTWithDropout(nn.Module):
    """Same model interface and base architecture as ``combined_model``."""

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config["context"]
        width = config["width"]
        query_heads = config["heads"]
        kv_heads = config.get("kv_heads", max(1, query_heads // 2))
        resid_dropout = float(config.get("resid_dropout", 0.0))
        embedding_dropout = float(config.get("embedding_dropout", 0.0))
        ffn_hidden_multiplier = float(config.get("ffn_hidden_multiplier", 8.0 / 3.0))
        if not 0.0 <= resid_dropout < 1.0:
            raise ValueError("resid_dropout must be in [0, 1)")
        if not 0.0 <= embedding_dropout < 1.0:
            raise ValueError("embedding_dropout must be in [0, 1)")
        if width % query_heads or (width // query_heads) % 2:
            raise ValueError("RoPE requires an even head width and width divisible by heads")
        if query_heads % kv_heads:
            raise ValueError("heads must be divisible by kv_heads")

        self.token = nn.Embedding(config["vocab"], width)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.blocks = nn.ModuleList([
            DropoutTransformerBlock(width, query_heads, kv_heads, resid_dropout,
                                    ffn_hidden_multiplier)
            for _ in range(config["depth"])
        ])
        self.final_norm = RMSNorm(width)
        self.lm_head = nn.Linear(width, config["vocab"], bias=False)
        self.apply(CombinedGPT._initialize)
        self.lm_head.weight = self.token.weight

    def forward(self, ids):
        if ids.shape[1] > self.context:
            raise ValueError(f"input length exceeds configured context {self.context}")
        x = self.embedding_dropout(self.token(ids))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return CombinedGPTWithDropout(config)
