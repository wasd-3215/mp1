"""Multi-query attention: all query heads share one key/value head."""
from .core import FrontierGPT


def build_model(config):
    return FrontierGPT(config, kind="mqa", kv_heads=1)
