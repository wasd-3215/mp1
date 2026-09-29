"""Shared, small GPT components used by the context-use experiments."""
import torch
from torch import nn
from torch.nn import functional as F


class ContextBlock(nn.Module):
    """Pre-norm transformer block with configurable causal attention policy."""

    def __init__(self, width, heads, attention_kind="global", local_window=64,
                 rotary=False):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        if rotary and self.head_width % 2:
            raise ValueError("rotary attention requires an even per-head width")
        if attention_kind not in ("global", "local"):
            raise ValueError("attention_kind must be 'global' or 'local'")
        if local_window < 1:
            raise ValueError("local_window must be positive")
        self.attention_kind = attention_kind
        self.local_window = local_window
        self.rotary = rotary
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))

    @staticmethod
    def _apply_rotary(q, k):
        """Rotate query/key pairs using token positions (no future information)."""
        length, pair_count = q.shape[-2], q.shape[-1] // 2
        positions = torch.arange(length, device=q.device, dtype=torch.float32)
        inv_freq = 10000.0 ** (-torch.arange(pair_count, device=q.device,
                                             dtype=torch.float32) / pair_count)
        angles = positions[:, None] * inv_freq[None, :]
        cos, sin = angles.cos().to(q.dtype)[None, None], angles.sin().to(q.dtype)[None, None]

        def rotate(x):
            even, odd = x[..., 0::2], x[..., 1::2]
            return torch.stack((even * cos - odd * sin,
                                even * sin + odd * cos), dim=-1).flatten(-2)

        return rotate(q), rotate(k)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, self.head_width
        ).permute(2, 0, 3, 1, 4)
        if self.rotary:
            q, k = self._apply_rotary(q, k)

        # Boolean SDPA masks mark allowed key positions. Both restrictions are
        # applied here so a local block can never see a future token.
        if self.attention_kind == "global":
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            query_pos = torch.arange(length, device=x.device)[:, None]
            key_pos = torch.arange(length, device=x.device)[None, :]
            allowed = (key_pos <= query_pos) & (key_pos > query_pos - self.local_window)
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed[None, None], is_causal=False
            )
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class ContextGPT(nn.Module):
    """GPT-shaped model with selectable position and attention mechanisms."""

    def __init__(self, config, position_kind="learned", attention_pattern=None,
                 local_window=64):
        super().__init__()
        if position_kind not in ("learned", "none", "rotary"):
            raise ValueError("position_kind must be learned, none, or rotary")
        self.config = dict(config)
        self.context = config["context"]
        width, depth = config["width"], config["depth"]
        if attention_pattern is None:
            attention_pattern = ["global"] * depth
        if len(attention_pattern) != depth:
            raise ValueError("attention_pattern must contain one entry per block")
        self.position_kind = position_kind
        self.token = nn.Embedding(config["vocab"], width)
        self.pos = (nn.Embedding(self.context, width)
                    if position_kind == "learned" else None)
        self.blocks = nn.ModuleList([
            ContextBlock(width, config["heads"], kind, local_window,
                         rotary=(position_kind == "rotary"))
            for kind in attention_pattern
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
        """Return unnormalized next-token logits with shape [batch,time,vocab]."""
        if ids.shape[1] > self.context:
            raise ValueError(f"input length exceeds configured context {self.context}")
        x = self.token(ids)
        if self.pos is not None:
            positions = torch.arange(ids.shape[1], device=ids.device)
            x = x + self.pos(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def predict_log_probs(self, ids):
        """Return normalized log probabilities for causal evaluation."""
        return F.log_softmax(self(ids).float(), dim=-1)
