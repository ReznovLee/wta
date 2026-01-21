"""Transformer subpackage for WTA-ConTra models.
Contains spatial-temporal encoders, geometric attention layers, and pointer-style decoders.
"""

from .spatial_temporal_encoder import SpatialTemporalEncoder
from .geometric_attention import GeometricAttention
from .pointer_decoder import PointerDecoder

__all__ = [
    "SpatialTemporalEncoder",
    "GeometricAttention",
    "PointerDecoder",
]