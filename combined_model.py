"""Compact GPT combining Pre-RMSNorm, RoPE, GQA, SwiGLU, and bias-free linears.

Select this implementation from ``code/`` with:
    python train.py --implementation combined_model --run-dir runs/myruns/combined

The public model interface matches the supplied trainer and evaluator.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """RMS normalization with a learned scale and no additive bias."""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(dtype=x.dtype)) * self.weight


class RotaryEmbedding(nn.Module):
    """Apply rotary positional features to query/key pairs along the time axis."""

    def __init__(self, head_width, base=10000.0):
        super().__init__()
        if head_width % 2:
            raise ValueError("RoPE requires an even attention head width")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_width, 2).float() / head_width))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q, k):
        length = q.shape[-2]
        positions = torch.arange(length, device=q.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq)
        cos = angles.cos().to(dtype=q.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=q.dtype)[None, None, :, :]

        def rotate(x):
            even, odd = x[..., 0::2], x[..., 1::2]
            return torch.stack((even * cos - odd * sin,
                                even * sin + odd * cos), dim=-1).flatten(-2)

        return rotate(q), rotate(k)


class GQARoPEAttention(nn.Module):
    """Causal grouped-query attention with RoPE and bias-free projections."""

    def __init__(self, width, query_heads, kv_heads):
        super().__init__()
        if width % query_heads:
            raise ValueError("width must be divisible by query_heads")
        if query_heads % kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_width = width // query_heads
        self.group_size = query_heads // kv_heads
        self.q_proj = nn.Linear(width, query_heads * self.head_width, bias=False)
        self.k_proj = nn.Linear(width, kv_heads * self.head_width, bias=False)
        self.v_proj = nn.Linear(width, kv_heads * self.head_width, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.rope = RotaryEmbedding(self.head_width)

    def forward(self, x):
        batch, length, width = x.shape
        q = self.q_proj(x).view(
            batch, length, self.query_heads, self.head_width
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch, length, self.kv_heads, self.head_width
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch, length, self.kv_heads, self.head_width
        ).transpose(1, 2)
        q, k = self.rope(q, k)

        # Portable PyTorch reference path: expand shared KV heads for SDPA.
        # This preserves GQA parameterization, though it may not realize
        # production-kernel KV-cache bandwidth savings during this benchmark.
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        merged = attended.transpose(1, 2).reshape(batch, length, width)
        return self.out_proj(merged)


class SwiGLU(nn.Module):
    """Parameter-matched SwiGLU feed-forward network without linear biases."""

    def __init__(self, width, multiple=8):
        super().__init__()
        # Three matrices (gate, value, down) use about 3*d*hidden parameters;
        # hidden ~= 8/3*d roughly matches a two-matrix 4*d GELU MLP.
        hidden = math.ceil((8 * width / 3) / multiple) * multiple
        self.gate_proj = nn.Linear(width, hidden, bias=False)
        self.value_proj = nn.Linear(width, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.value_proj(x))


class TransformerBlock(nn.Module):
    """Pre-RMSNorm attention and SwiGLU block with residual connections."""

    def __init__(self, width, query_heads, kv_heads):
        super().__init__()
        self.attn_norm = RMSNorm(width)
        self.attention = GQARoPEAttention(width, query_heads, kv_heads)
        self.ffn_norm = RMSNorm(width)
        self.ffn = SwiGLU(width)

    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class CombinedGPT(nn.Module):
    """Language model exposing the interfaces required by MP1."""

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config["context"]
        width = config["width"]
        query_heads = config["heads"]
        # Default to half as many KV heads as query heads (GQA); an optional
        # config key makes the grouping explicit for controlled variants.
        kv_heads = config.get("kv_heads", max(1, query_heads // 2))
        if width % query_heads or (width // query_heads) % 2:
            raise ValueError("RoPE requires an even head width and width divisible by heads")
        if query_heads % kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        self.token = nn.Embedding(config["vocab"], width)
        self.blocks = nn.ModuleList([
            TransformerBlock(width, query_heads, kv_heads)
            for _ in range(config["depth"])
        ])
        self.final_norm = RMSNorm(width)
        self.lm_head = nn.Linear(width, config["vocab"], bias=False)
        self.apply(self._initialize)
        # Weight tying avoids duplicating the token embedding in the output head.
        self.lm_head.weight = self.token.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, ids):
        """Return unnormalized next-token logits: [batch, time, vocab]."""
        if ids.shape[1] > self.context:
            raise ValueError(f"input length exceeds configured context {self.context}")
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    def predict_log_probs(self, ids):
        """Return normalized natural-log probabilities for causal evaluation."""
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    """Factory loaded by train.py/evaluate.py for --implementation combined_model."""
    return CombinedGPT(config)
