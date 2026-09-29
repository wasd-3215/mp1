"""Grouped-query attention: four query heads share two key/value heads."""
from .core import FrontierGPT


def build_model(config):
    heads = config["heads"]
    if heads < 2 or heads % 2:
        raise ValueError("GQA requires an even number of at least two query heads")
    return FrontierGPT(config, kind="gqa", kv_heads=heads // 2)
