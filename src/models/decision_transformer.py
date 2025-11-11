import torch
import torch.nn as nn
from .transformer.spatial_temporal_encoder import SpatialTemporalEncoder
from .transformer.pointer_decoder import PointerDecoder


class DecisionTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dt_cfg = cfg.get('decision_transformer', {})
        d_model = dt_cfg.get('d_model', 256)
        n_heads = dt_cfg.get('n_heads', 8)
        n_layers = dt_cfg.get('n_layers', 4)
        dropout = dt_cfg.get('dropout', 0.1)
        self.encoder = SpatialTemporalEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.decoder = PointerDecoder(d_model=d_model)
        self.top_k = dt_cfg.get('pointer_top_k', 32)

    def forward(self, states, returns_to_go, action_masks):
        enc = self.encoder(states)
        assignments = self.decoder.decode(enc, enc, action_masks, top_k=self.top_k)
        return assignments

    def loss_offline(self, batch):
        return {'loss': torch.tensor(0.0)}

    def sample_action(self, obs, masks):
        am = masks['ammo_mask'] & masks['range_mask'] & masks['assign_mask']
        idx = torch.nonzero(torch.tensor(am))
        if idx.numel() == 0:
            return None
        i, j = idx[0].tolist()
        return (i, j)