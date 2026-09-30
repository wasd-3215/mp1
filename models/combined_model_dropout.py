"""Compatibility export for the canonical dropout-capable decoder."""
from .full_attention_dropout import (
    CombinedGPTWithDropout, ConfigurableSwiGLU, DropoutTransformerBlock,
    build_model,
)
