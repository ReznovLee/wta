"""
Geometric Attention layer for WTA-ConTra.

Augments dot-product attention with distance/time-window aware biases.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricAttention(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None):
        """
        Args:
            query, key, value: [B, L, D]
            attn_mask: [B, L, S] mask with True to mask out
            bias: [B, L, S] additive bias (e.g., distance/time penalties)
        Returns:
            attended context and attention weights.
        """
        if bias is not None:
            # MultiheadAttention in PyTorch accepts attn_mask as additive mask when float
            if attn_mask is None:
                attn_mask = bias
            else:
                attn_mask = attn_mask.to(bias.dtype) + bias
        out, w = self.attn(query, key, value, attn_mask=attn_mask)
        return out, w