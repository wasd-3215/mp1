"""Learned per-token blend of local and full-prefix attention contexts."""
from .core import AttentionGPT


def build_model(config):
    return AttentionGPT(
        config,
        pattern=["gated_local_global"] * config["depth"],
        local_window=config.get("local_window", 64),
    )
