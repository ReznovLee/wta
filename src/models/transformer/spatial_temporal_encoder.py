import torch
import torch.nn as nn


class SpatialTemporalEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, states, mask=None, src_key_padding_mask=None):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        
        # nn.TransformerEncoder 期望 mask 形状为 (S, S) 或 (N*S, S)
        # 其中 S 是序列长度。
        # 如果 mask 是布尔型，True 表示允许参与注意力？不，PyTorch 中 True 通常表示被 Mask 掉（忽略），或者相反，取决于版本和 API。
        # nn.TransformerEncoderLayer 的 forward 参数 src_mask:
        # "If a BoolTensor is provided, positions with the value True are not allowed to attend while False values will be unchanged."
        # 也就是 True = Masked (Ignore)。
        # 但也有可能是 attn_mask (Float), 0 = OK, -inf = Masked.
        
        # 为了安全，建议使用 Float Mask (0/-inf) 或根据 PyTorch 文档。
        # 这里直接透传 mask。
        
        return self.encoder(states, mask=mask, src_key_padding_mask=src_key_padding_mask)
