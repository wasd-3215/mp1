"""Gated attention with Q/K normalization and partial rotary positions."""
from .core import FrontierGPT


def build_model(config):
    return FrontierGPT(config, kind="gated")
