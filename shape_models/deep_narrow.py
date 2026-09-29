"""Deep/narrow shape: width 96 and eight transformer blocks."""
from combined_model import CombinedGPT


def build_model(config):
    if (config["width"], config["depth"]) != (96, 8):
        raise ValueError("deep_narrow requires width=96 and depth=8")
    return CombinedGPT(config)
