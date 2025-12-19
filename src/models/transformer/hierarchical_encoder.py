import torch
import torch.nn as nn


class HierarchicalEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        enc_layer_t = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        enc_layer_i = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.targets_encoder = nn.TransformerEncoder(enc_layer_t, num_layers=n_layers)
        self.interceptors_encoder = nn.TransformerEncoder(enc_layer_i, num_layers=n_layers)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, targets_seq, interceptors_seq, global_state):
        if not isinstance(targets_seq, torch.Tensor):
            targets_seq = torch.tensor(targets_seq, dtype=torch.float32)
        if not isinstance(interceptors_seq, torch.Tensor):
            interceptors_seq = torch.tensor(interceptors_seq, dtype=torch.float32)
        if not isinstance(global_state, torch.Tensor):
            global_state = torch.tensor(global_state, dtype=torch.float32)
        t_enc = self.targets_encoder(targets_seq)
        i_enc = self.interceptors_encoder(interceptors_seq)
        q = i_enc
        k = t_enc
        v = t_enc
        attn_out, _ = self.cross_attn(q, k, v)
        g = global_state
        if g.dim() == 2:
            g = g.unsqueeze(1)
        gi = torch.cat([attn_out, g.expand_as(attn_out)], dim=-1)
        fused = self.fuse(gi)
        return fused
