"""Three gated-delta blocks followed by one exact full-attention block."""
from .core import FrontierGPT


def build_model(config):
    depth = config["depth"]
    if depth < 2:
        raise ValueError("the hybrid needs at least one delta and one full block")
    # A compact 3:1-style hybrid. Keep a full-attention layer for exact retrieval.
    pattern = ["delta"] * (depth - 1) + ["gated"]
    return FrontierGPT(config, pattern=pattern)
