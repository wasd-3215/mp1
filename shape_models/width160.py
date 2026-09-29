"""Width experiment: four layers at width 160 with four Q / two KV heads."""
from combined_model import CombinedGPT


def build_model(config):
    if (config["width"], config["depth"], config["heads"], config["kv_heads"]) != (160, 4, 4, 2):
        raise ValueError("width160 requires width=160, depth=4, heads=4, kv_heads=2")
    return CombinedGPT(config)
