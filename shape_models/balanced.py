"""Reference shape: width 128 and four transformer blocks."""
from combined_model import CombinedGPT


def build_model(config):
    if (config["width"], config["depth"]) != (128, 4):
        raise ValueError("balanced requires width=128 and depth=4")
    return CombinedGPT(config)
