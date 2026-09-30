"""Ablation: omit explicit position embeddings; retain full causal attention."""
from .core import ContextGPT


def build_model(config):
    return ContextGPT(config, position_kind="none")
