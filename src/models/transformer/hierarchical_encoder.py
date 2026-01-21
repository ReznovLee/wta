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

    def forward(self, targets_seq, interceptors_seq, global_state, t_mask=None, i_mask=None, cross_mask=None, t_padding_mask=None, i_padding_mask=None):
        if not isinstance(targets_seq, torch.Tensor):
            targets_seq = torch.tensor(targets_seq, dtype=torch.float32)
        if not isinstance(interceptors_seq, torch.Tensor):
            interceptors_seq = torch.tensor(interceptors_seq, dtype=torch.float32)
        if not isinstance(global_state, torch.Tensor):
            global_state = torch.tensor(global_state, dtype=torch.float32)
        
        # 独立编码 (支持掩码)
        t_enc = self.targets_encoder(targets_seq, mask=t_mask, src_key_padding_mask=t_padding_mask)
        i_enc = self.interceptors_encoder(interceptors_seq, mask=i_mask, src_key_padding_mask=i_padding_mask)
        
        # 交互与融合 (Cross Attention)
        # Query = Interceptors, Key = Targets, Value = Targets
        # 这里的物理意义是：每个拦截器去关注所有目标
        q = i_enc
        k = t_enc
        v = t_enc
        
        # Cross Attention 是否需要 Mask？
        # 如果是帧内 (Frame-wise) Attention，不需要 Mask（假设 Key 已经包含了所需的历史信息）。
        # 如果是全序列 Attention，则可能需要 key_padding_mask 或 attn_mask。
        # 考虑到 targets_encoder 已经处理了时间因果性，t_enc[t] 已经聚合了 targets[0...t]。
        # 如果我们希望 interceptor[t] 只关注 targets[t] (及其历史)，
        # 那么如果是对齐的序列 (B, Seq, D)，MultiheadAttention 默认是全关注 (all-to-all) 除非提供 mask。
        # 但通常 Transformer Decoder 中的 Cross Attention 不加 Causal Mask，因为 Key 是完整的 Encoder Output。
        # 在这里，Target 和 Interceptor 是同步演进的。
        # 如果我们把它们看作并行序列，interceptor[t] 应该能看到 target[0...t]。
        # 实际上，如果 t_enc[t] 已经包含了 0...t 的信息，那么 interceptor[t] 只需要关注 t_enc[t] 吗？
        # 不，通常 Attention 允许关注整个 Memory。
        # 但是，如果是 Online Decision，interceptor[t] 不能看到 target[t+1...]。
        # 所以 Cross Attention 也需要 Causal Mask！
        # 除非我们是 Frame-wise 处理的（Seq=1）。
        # 如果输入是序列 (B, S, D)，我们需要一个 S x S 的 Mask 保证 q[t] 不关注 k[t+1:]。
        
        # 假设 t_mask 和 i_mask 已经是 S x S 的因果掩码。
        # 我们可以复用 t_mask 作为 attn_mask (假设 shape 兼容)。
        # MultiheadAttention forward(query, key, value, attn_mask=...)
        # attn_mask shape: (L, S) where L is target seq len, S is source seq len.
        # 这里 L=S.
        
        attn_out, _ = self.cross_attn(q, k, v, attn_mask=cross_mask, key_padding_mask=t_padding_mask)
        
        # 融合全局状态
        g = global_state
        if g.dim() == 2:
            g = g.unsqueeze(1)
        
        # 将全局状态扩展拼接到每个拦截器的表示上
        gi = torch.cat([attn_out, g.expand_as(attn_out)], dim=-1)
        fused = self.fuse(gi)
        
        # 返回：融合后的特征（用于后续决策或Value预测），以及独立的目标/拦截器特征（用于Pointer Decoder的注意力计算）
        return fused, t_enc, i_enc
