import torch
import torch.nn as nn


class SpatialTemporalEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.randn(1, 1, 256)
        return self.encoder(states)