"""Causal hybrid GPT using alternating attention and depthwise-convolution blocks."""
import torch
from torch import nn
from torch.nn import functional as F

from .transformer_components import CombinedGPT, GQARoPEAttention, RMSNorm
from .full_attention_dropout import ConfigurableSwiGLU


class CausalConvMixer(nn.Module):
    """Depthwise causal convolution followed by channel mixing."""

    def __init__(self, width, kernel_size):
        super().__init__()
        if kernel_size < 2:
            raise ValueError("conv_kernel_size must be at least 2")
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv1d(width, width, kernel_size,
                                   groups=width, bias=False)
        nn.init.normal_(self.depthwise.weight, mean=0.0, std=0.02)
        self.channel_mix = nn.Linear(width, width, bias=False)

    def forward(self, x):
        # Left-only padding ensures position t can see no tokens after t.
        channels_first = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        local = self.depthwise(channels_first).transpose(1, 2)
        return self.channel_mix(F.silu(local))


class HybridBlock(nn.Module):
    def __init__(self, width, mixer, resid_dropout, ffn_hidden_multiplier):
        super().__init__()
        self.mixer_norm = RMSNorm(width)
        self.mixer = mixer
        self.mixer_dropout = nn.Dropout(resid_dropout)
        self.ffn_norm = RMSNorm(width)
        self.ffn = ConfigurableSwiGLU(width, ffn_hidden_multiplier)
        self.ffn_dropout = nn.Dropout(resid_dropout)

    def forward(self, x):
        x = x + self.mixer_dropout(self.mixer(self.mixer_norm(x)))
        return x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))


class HybridGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config["context"]
        width = config["width"]
        heads = config["heads"]
        kv_heads = config.get("kv_heads", max(1, heads // 2))
        depth = int(config["depth"])
        attention_layers = set(config.get("attention_layers", [0, 2]))
        kernel_size = int(config.get("conv_kernel_size", 5))
        resid_dropout = float(config.get("resid_dropout", 0.0))
        embedding_dropout = float(config.get("embedding_dropout", 0.0))
        ffn_multiplier = float(config.get("ffn_hidden_multiplier", 8.0 / 3.0))

        if depth < 1 or not attention_layers or any(
                index < 0 or index >= depth for index in attention_layers):
            raise ValueError("attention_layers must contain valid layer indices")
        if width % heads or (width // heads) % 2:
            raise ValueError("RoPE requires an even head width and width divisible by heads")
        if heads % kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        if not 0.0 <= resid_dropout < 1.0 or not 0.0 <= embedding_dropout < 1.0:
            raise ValueError("dropout probabilities must be in [0, 1)")
        if ffn_multiplier <= 0:
            raise ValueError("ffn_hidden_multiplier must be positive")

        self.token = nn.Embedding(config["vocab"], width)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        blocks = []
        for index in range(depth):
            if index in attention_layers:
                mixer = GQARoPEAttention(width, heads, kv_heads)
            else:
                mixer = CausalConvMixer(width, kernel_size)
            blocks.append(HybridBlock(width, mixer, resid_dropout, ffn_multiplier))
        self.blocks = nn.ModuleList(blocks)
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
    return HybridGPT(config)
