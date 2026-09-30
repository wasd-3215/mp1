"""Compatibility export; use models.full_attention_dropout for new runs."""
from .transformer_components import (
    CombinedGPT, GQARoPEAttention, RMSNorm, RotaryEmbedding, SwiGLU,
    TransformerBlock, build_model,
)
