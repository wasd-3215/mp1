"""RoPE plus alternating local/full-prefix causal attention blocks."""
from .core import AttentionGPT


def build_model(config):
    # Even blocks cover recent context; odd blocks can retrieve the full prefix.
    pattern = ["local" if index % 2 == 0 else "global"
               for index in range(config["depth"])]
    return AttentionGPT(
        config,
        pattern=pattern,
        rotary=True,
        local_window=config.get("local_window", 64),
        learned_positions=False,
    )
