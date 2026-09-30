"""Lecture-inspired learned top-k sparse attention (dense reference kernel)."""
from .core import FrontierGPT


def build_model(config):
    return FrontierGPT(config, kind="dsa", top_k=config.get("top_k", 32))
