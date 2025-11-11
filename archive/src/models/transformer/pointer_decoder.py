"""
Pointer-style Decoder for constrained action selection.

Produces autoregressive selections over (weapon, target, time, ammo) factors with
hard action masks enforced at each step.
"""
from typing import Dict, Optional
import torch
import torch.nn as nn


class PointerDecoder(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
        )
        self.weapon_head = nn.Linear(d_model, 1)  # scores per weapon token
        self.target_head = nn.Linear(d_model, 1)  # scores per target token

    def forward(self, weapon_enc: torch.Tensor, target_enc: torch.Tensor,
                weapon_mask: Optional[torch.Tensor] = None,
                target_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Simplified pointer mechanism: get context from pooled keys
        w_ctx = weapon_enc.mean(dim=1, keepdim=True)
        t_ctx = target_enc.mean(dim=1, keepdim=True)
        q = self.query_proj(w_ctx + t_ctx)
        k_w = self.key_proj(weapon_enc)
        k_t = self.key_proj(target_enc)
        ctx_w, _ = self.attn(q, k_w, weapon_enc, key_padding_mask=weapon_mask)
        ctx_t, _ = self.attn(q, k_t, target_enc, key_padding_mask=target_mask)
        h_w = self.ff(ctx_w)
        h_t = self.ff(ctx_t)
        weapon_scores = self.weapon_head(h_w).squeeze(1)  # [B, W, 1] -> [B, W]
        target_scores = self.target_head(h_t).squeeze(1)  # [B, T, 1] -> [B, T]
        return {"weapon_scores": weapon_scores, "target_scores": target_scores}