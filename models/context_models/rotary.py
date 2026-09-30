"""Position experiment: use rotary relative-position features in attention."""
from .core import ContextGPT


def build_model(config):
    return ContextGPT(config, position_kind="rotary")
