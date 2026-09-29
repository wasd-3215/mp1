"""Small causal attention variants for the fixed-context MP1 benchmark."""
import torch
from torch import nn
from torch.nn import functional as F


def _rotary_prefix(x):
    """Apply RoPE to the first half of each head; leave the rest unchanged."""
    rotary_width = (x.shape[-1] // 2) // 2 * 2
    if rotary_width == 0:
        return x
    length = x.shape[-2]
    pair_count = rotary_width // 2
    pos = torch.arange(length, device=x.device, dtype=torch.float32)
    inv_freq = 10000.0 ** (-torch.arange(pair_count, device=x.device,
                                         dtype=torch.float32) / pair_count)
    angle = pos[:, None] * inv_freq[None, :]
    cos = angle.cos().to(x.dtype)[None, None]
    sin = angle.sin().to(x.dtype)[None, None]
    part = x[..., :rotary_width]
    even, odd = part[..., 0::2], part[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin,
                           even * sin + odd * cos), dim=-1).flatten(-2)
    return torch.cat((rotated, x[..., rotary_width:]), dim=-1)


def _causal_local_mask(length, window, device):
    qpos = torch.arange(length, device=device)[:, None]
    kpos = torch.arange(length, device=device)[None, :]
    return (kpos <= qpos) & (kpos > qpos - window)


class FrontierBlock(nn.Module):
    """Pre-norm block with lecture-inspired attention or sequence mixing."""

    def __init__(self, width, heads, context, kind="gqa", kv_heads=None,
                 top_k=32, index_dim=8):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.context = context
        self.kind = kind
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        if kind in ("gqa", "mqa"):
            if kv_heads is None or heads % kv_heads:
                raise ValueError("query heads must be divisible by KV heads")
            self.kv_heads = kv_heads
            self.q_proj = nn.Linear(width, width)
            self.k_proj = nn.Linear(width, kv_heads * self.head_width)
            self.v_proj = nn.Linear(width, kv_heads * self.head_width)
        else:
            self.qkv = nn.Linear(width, 3 * width)
            self.kv_heads = heads
        if kind == "gated":
            self.output_gate = nn.Linear(width, width)
            self.q_norm = nn.LayerNorm(self.head_width, elementwise_affine=False)
            self.k_norm = nn.LayerNorm(self.head_width, elementwise_affine=False)
        elif kind == "dsa":
            if not 1 <= top_k <= context:
                raise ValueError("top_k must be between 1 and context")
            self.top_k = top_k
            self.index_q = nn.Linear(width, heads * index_dim, bias=False)
            self.index_k = nn.Linear(width, heads * index_dim, bias=False)
            self.index_dim = index_dim
        elif kind == "delta":
            self.alpha = nn.Linear(width, heads)
            self.beta = nn.Linear(width, heads)
        self.out_proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))

    def _project_qkv(self, normalized):
        batch, length, _ = normalized.shape
        if self.kind in ("gqa", "mqa"):
            q = self.q_proj(normalized).view(
                batch, length, self.heads, self.head_width).transpose(1, 2)
            k = self.k_proj(normalized).view(
                batch, length, self.kv_heads, self.head_width).transpose(1, 2)
            v = self.v_proj(normalized).view(
                batch, length, self.kv_heads, self.head_width).transpose(1, 2)
            # Portable reference path; optimized kernels may broadcast KV heads.
            repeat = self.heads // self.kv_heads
            return q, k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        q, k, v = self.qkv(normalized).view(
            batch, length, 3, self.heads, self.head_width
        ).permute(2, 0, 3, 1, 4)
        return q, k, v

    def _dsa_attention(self, normalized, q, k, v):
        """Top-k learned candidate selection with a causal attention mask.

        The small indexer ranks keys. Its selected scores also enter the final
        softmax, giving the selector gradients for selected candidates. This
        classroom-scale implementation uses dense score tensors; it does not
        promise sparse-kernel speedups.
        """
        batch, length, _ = normalized.shape
        iq = self.index_q(normalized).view(
            batch, length, self.heads, self.index_dim).transpose(1, 2)
        ik = self.index_k(normalized).view(
            batch, length, self.heads, self.index_dim).transpose(1, 2)
        index_scores = torch.matmul(iq, ik.transpose(-2, -1)) / (self.index_dim ** 0.5)
        qpos = torch.arange(length, device=normalized.device)[:, None]
        kpos = torch.arange(length, device=normalized.device)[None, :]
        causal = kpos <= qpos
        index_scores = index_scores.masked_fill(~causal[None, None], float("-inf"))
        # At early positions, top-k can include masked future indices; the final
        # causal intersection below still prevents those keys from being used.
        selected = index_scores.topk(min(self.top_k, length), dim=-1).indices
        keep = torch.zeros_like(index_scores, dtype=torch.bool).scatter_(-1, selected, True)
        keep = keep & causal[None, None]
        bias = index_scores.masked_fill(~keep, float("-inf"))
        return F.scaled_dot_product_attention(q, k, v,
                                              attn_mask=bias.to(q.dtype),
                                              is_causal=False)

    def _delta_attention(self, normalized, q, k, v):
        """Simple causal gated-delta recurrence; state resets on every call."""
        batch, heads, length, head_width = q.shape
        alpha = self.alpha(normalized).sigmoid().transpose(1, 2).unsqueeze(-1)
        beta = self.beta(normalized).sigmoid().transpose(1, 2).unsqueeze(-1)
        state = q.new_zeros(batch, heads, head_width, head_width)
        outputs = []
        for t in range(length):
            qt = F.normalize(q[:, :, t], dim=-1)
            kt = F.normalize(k[:, :, t], dim=-1)
            vt = v[:, :, t]
            prediction = torch.einsum("bhvd,bhd->bhv", state, kt)
            correction = vt - prediction
            state = (alpha[:, :, t, :] * state +
                     beta[:, :, t, :] * correction.unsqueeze(-1) * kt.unsqueeze(-2))
            outputs.append(torch.einsum("bhvd,bhd->bhv", state, qt))
        return torch.stack(outputs, dim=2)

    def forward(self, x):
        batch, length, width = x.shape
        normalized = self.norm1(x)
        q, k, v = self._project_qkv(normalized)
        if self.kind in ("gqa", "mqa"):
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.kind == "gated":
            # Q/K normalization controls score scale; partial RoPE leaves part
            # of each head available for position-agnostic content matching.
            q = _rotary_prefix(self.q_norm(q))
            k = _rotary_prefix(self.k_norm(k))
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attended = attended * self.output_gate(normalized).sigmoid().view(
                batch, length, self.heads, self.head_width).transpose(1, 2)
        elif self.kind == "dsa":
            attended = self._dsa_attention(normalized, q, k, v)
        elif self.kind == "delta":
            attended = self._delta_attention(normalized, q, k, v)
        else:
            raise ValueError(f"unsupported attention kind: {self.kind}")
        merged = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + self.out_proj(merged)
        return x + self.mlp(self.norm2(x))


class FrontierGPT(nn.Module):
    """GPT interface used by each independent lecture-mechanism module."""

    def __init__(self, config, kind="gqa", kv_heads=None, pattern=None,
                 top_k=32):
        super().__init__()
        self.config = dict(config)
        self.context = config["context"]
        width, depth = config["width"], config["depth"]
        if pattern is None:
            pattern = [kind] * depth
        if len(pattern) != depth:
            raise ValueError("pattern must contain one kind per transformer block")
        self.token = nn.Embedding(config["vocab"], width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList([
            FrontierBlock(width, config["heads"], self.context, block_kind,
                          kv_heads=kv_heads, top_k=top_k)
            for block_kind in pattern
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
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token(ids) + self.pos(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)
