"""GPT model with cyclic inter-layer parameter sharing and residual dropout.

The number of execution stages is ``depth``.  ``shared_layers`` distinct
Transformer blocks are reused cyclically across those stages.  For example,
depth=4 and shared_layers=2 executes blocks 0, 1, 0, 1.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .transformer_components import CombinedGPT, GQARoPEAttention, RMSNorm
from .full_attention_dropout import ConfigurableSwiGLU


class SharedTransformerBlock(nn.Module):
    def __init__(self, width, query_heads, kv_heads, resid_dropout,
                 ffn_hidden_multiplier):
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


class SharedGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config["context"]
        width = config["width"]
        heads = config["heads"]
        kv_heads = config.get("kv_heads", max(1, heads // 2))
        depth = int(config["depth"])
        shared_layers = int(config.get("shared_layers", 1))
        resid_dropout = float(config.get("resid_dropout", 0.0))
        embedding_dropout = float(config.get("embedding_dropout", 0.0))
        ffn_multiplier = float(config.get("ffn_hidden_multiplier", 8.0 / 3.0))

        if depth < 1 or not 1 <= shared_layers <= depth:
            raise ValueError("shared_layers must be between 1 and depth")
        if width % heads or (width // heads) % 2:
            raise ValueError("RoPE requires an even head width and width divisible by heads")
        if heads % kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        if not 0.0 <= resid_dropout < 1.0 or not 0.0 <= embedding_dropout < 1.0:
            raise ValueError("dropout probabilities must be in [0, 1)")
        if ffn_multiplier <= 0:
            raise ValueError("ffn_hidden_multiplier must be positive")

        self.depth = depth
        self.shared_layers = shared_layers
        self.token = nn.Embedding(config["vocab"], width)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.blocks = nn.ModuleList([
            SharedTransformerBlock(width, heads, kv_heads, resid_dropout,
                                   ffn_multiplier)
            for _ in range(shared_layers)
        ])
        self.final_norm = RMSNorm(width)
        self.lm_head = nn.Linear(width, config["vocab"], bias=False)
        self.apply(CombinedGPT._initialize)
        self.lm_head.weight = self.token.weight

    def forward(self, ids):
        if ids.shape[1] > self.context:
            raise ValueError(f"input length exceeds configured context {self.context}")
        x = self.embedding_dropout(self.token(ids))
        for layer_index in range(self.depth):
            x = self.blocks[layer_index % self.shared_layers](x)
        return self.lm_head(self.final_norm(x))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return SharedGPT(config)
