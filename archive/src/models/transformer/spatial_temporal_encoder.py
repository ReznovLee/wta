"""
Spatial-Temporal Encoder for WTA-ConTra.

Encodes two unordered sets (weapons and targets) with cross-attention,
positional-temporal embeddings, and capacity/cooldown attributes into
fixed-length representations for downstream decoding.
"""
from typing import Dict, Any
import torch
import torch.nn as nn


class SpatialTemporalEncoder(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.weapon_proj = nn.Linear(d_model, d_model)
        self.target_proj = nn.Linear(d_model, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                   dim_feedforward=4 * d_model,
                                                   dropout=dropout, batch_first=True)
        self.weapon_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.target_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, weapon_feats: torch.Tensor, target_feats: torch.Tensor,
                weapon_mask: torch.Tensor = None, target_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            weapon_feats: [B, W, D]
            target_feats: [B, T, D]
            weapon_mask: [B, W] boolean mask where True indicates padding/invalid
            target_mask: [B, T] boolean mask where True indicates padding/invalid
        Returns:
            dict with encoded representations and pooled summaries.
        """
        w = self.weapon_proj(weapon_feats)
        t = self.target_proj(target_feats)

        w_enc = self.weapon_encoder(w, src_key_padding_mask=weapon_mask)
        t_enc = self.target_encoder(t, src_key_padding_mask=target_mask)

        w_pool = w_enc.mean(dim=1)
        t_pool = t_enc.mean(dim=1)
        return {"weapon_enc": w_enc, "target_enc": t_enc, "weapon_pool": w_pool, "target_pool": t_pool}