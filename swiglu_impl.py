"""GPT baseline with a parameter-matched SwiGLU feed-forward layer."""
import torch
from torch import nn
from torch.nn import functional as F

from model import Block, GPT


class SwiGLUBlock(Block):
    """Baseline attention block with a SwiGLU MLP in place of GELU MLP."""

    def __init__(self, width=128, heads=4):
        super().__init__(width, heads)
        # GELU MLP has about 8 * width * width hidden-weight parameters.
        # SwiGLU uses three width-by-hidden matrices, so hidden ~= 8/3 * width.
        hidden = round(8 * width / 3)
        self.mlp = nn.ModuleDict({
            'gate': nn.Linear(width, hidden),
            'value': nn.Linear(width, hidden),
            'down': nn.Linear(hidden, width),
        })
        self.apply(GPT.initialize)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))

        normalized = self.norm2(x)
        gated = F.silu(self.mlp['gate'](normalized)) * self.mlp['value'](normalized)
        return x + self.mlp['down'](gated)


class SwiGLUGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        width = config['width']
        self.blocks = nn.ModuleList(
            SwiGLUBlock(width, config['heads']) for _ in range(config['depth'])
        )


def build_model(config):
    return SwiGLUGPT(config)

# 训练指令和测试指令：
# python train.py --implementation swiglu_impl --run-dir runs/myruns/swiglu --seed 17 --steps 200