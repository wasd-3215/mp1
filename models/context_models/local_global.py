"""Attention experiment: alternate local and full-prefix causal blocks."""
from .core import ContextGPT


def build_model(config):
    # Even-numbered blocks are local; odd-numbered blocks can access the full
    # prefix. This keeps every layer causal while mixing context ranges.
    pattern = ["local" if index % 2 == 0 else "global"
               for index in range(config["depth"])]
    return ContextGPT(config, position_kind="learned",
                      attention_pattern=pattern,
                      local_window=config.get("local_window", 64))
