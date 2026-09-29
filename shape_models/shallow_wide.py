"""Shallow/wide shape: width 168 and two transformer blocks."""
from combined_model import CombinedGPT


def build_model(config):
    if (config["width"], config["depth"]) != (168, 2):
        raise ValueError("shallow_wide requires width=168 and depth=2")
    return CombinedGPT(config)
