"""Full causal attention with a learned per-head relative-distance bias."""
from .core import AttentionGPT


def build_model(config):
    return AttentionGPT(config, pattern=["relative"] * config["depth"])
