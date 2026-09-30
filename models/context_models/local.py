"""Attention experiment: restrict each block to a recent causal window."""
from .core import ContextGPT


def build_model(config):
    return ContextGPT(config, position_kind="learned",
                      attention_pattern=["local"] * config["depth"],
                      local_window=config.get("local_window", 64))
