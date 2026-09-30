"""Shared transformer code for attention-focused MP1 variants."""
import torch
from torch import nn
from torch.nn import functional as F


def apply_rotary(q, k):
    """Apply rotary position features to Q and K, preserving causal inputs."""
    length, pair_count = q.shape[-2], q.shape[-1] // 2
    positions = torch.arange(length, device=q.device, dtype=torch.float32)
    inv_freq = 10000.0 ** (-torch.arange(pair_count, device=q.device,
                                         dtype=torch.float32) / pair_count)
    angles = positions[:, None] * inv_freq[None, :]
    cos = angles.cos().to(q.dtype)[None, None]
    sin = angles.sin().to(q.dtype)[None, None]

    def rotate(x):
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin,
                            even * sin + odd * cos), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


class AttentionBlock(nn.Module):
    """Pre-norm transformer block with one of three causal attention designs."""

    def __init__(self, width, heads, context, kind="global", local_window=64,
                 rotary=False):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        if rotary and self.head_width % 2:
            raise ValueError("rotary attention requires even head width")
        if kind not in ("global", "relative", "gated_local_global", "local"):
            raise ValueError(f"unsupported attention kind: {kind}")
        if local_window < 1:
            raise ValueError("local_window must be positive")
        self.kind = kind
        self.local_window = local_window
        self.rotary = rotary
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))
        if kind == "relative":
            # One learnable additive attention bias per head and causal distance.
            self.distance_bias = nn.Parameter(torch.zeros(heads, context))
        else:
            self.register_parameter("distance_bias", None)
        if kind == "gated_local_global":
            # A token- and head-specific gate selects local vs full-prefix context.
            self.context_gate = nn.Linear(width, heads)
        else:
            self.context_gate = None

    def _local_mask(self, length, device):
        query = torch.arange(length, device=device)[:, None]
        key = torch.arange(length, device=device)[None, :]
        # Include the current token and at most local_window - 1 predecessors.
        return (key <= query) & (key > query - self.local_window)

    def forward(self, x):
        batch, length, width = x.shape
        normalized = self.norm1(x)
        q, k, v = self.qkv(normalized).view(
            batch, length, 3, self.heads, self.head_width
        ).permute(2, 0, 3, 1, 4)
        if self.rotary:
            q, k = apply_rotary(q, k)

        if self.kind == "relative":
            query = torch.arange(length, device=x.device)[:, None]
            key = torch.arange(length, device=x.device)[None, :]
            distance = (query - key).clamp(min=0, max=self.distance_bias.shape[1] - 1)
            bias = self.distance_bias[:, distance]
            bias = bias.masked_fill((key > query)[None], float("-inf"))
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias[None].to(q.dtype), is_causal=False
            )
        elif self.kind == "gated_local_global":
            global_context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            local_context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=self._local_mask(length, x.device)[None, None],
                is_causal=False
            )
            gate = self.context_gate(normalized).sigmoid().transpose(1, 2)[..., None]
            attended = gate * global_context + (1.0 - gate) * local_context
        elif self.kind == "local":
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=self._local_mask(length, x.device)[None, None],
                is_causal=False
            )
        else:
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class AttentionGPT(nn.Module):
    """GPT interface shared by the attention experiment modules."""

    def __init__(self, config, pattern=None, rotary=False,
                 local_window=64, learned_positions=True):
        super().__init__()
        self.config = dict(config)
        self.context = config["context"]
        width, depth = config["width"], config["depth"]
        if pattern is None:
            pattern = ["global"] * depth
        if len(pattern) != depth:
            raise ValueError("pattern must contain one attention kind per block")
        self.token = nn.Embedding(config["vocab"], width)
        self.pos = nn.Embedding(self.context, width) if learned_positions else None
        self.blocks = nn.ModuleList([
            AttentionBlock(width, config["heads"], self.context, kind,
                           local_window, rotary=rotary)
            for kind in pattern
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config["vocab"], bias=False)
        self.apply(self._initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        if ids.shape[1] > self.context:
            raise ValueError(f"input length exceeds context={self.context}")
        x = self.token(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)
