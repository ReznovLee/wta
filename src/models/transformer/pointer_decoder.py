import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PointerDecoder(nn.Module):
    """指针解码器（完整版）：
    - 使用 Scaled Dot-Product Attention 计算 Query（武器）与 Key（目标）的匹配分数。
    - 结合 TTI（Time-to-Intercept）作为物理约束偏置。
    - 支持 Masking 以处理无效分配。
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        # alpha 用于控制 TTI 对最终分数的负向影响权重 (Score = Attn - alpha * TTI + bias)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def get_combined_mask(self, action_masks):
        """辅助函数：合并各种约束掩码"""
        masks = action_masks or {}
        
        def get_mask(name):
            m = masks.get(name)
            if m is None:
                return None
            if isinstance(m, torch.Tensor):
                return m.bool()
            else:
                return torch.tensor(m, dtype=torch.bool)

        mask_names = ['assign_mask', 'ammo_mask', 'range_mask', 'alive_mask', 'capacity_mask', 'defense_time_mask']
        combined = None
        for mn in mask_names:
            m = get_mask(mn)
            if m is None:
                continue
            # 确保掩码在同一设备
            if combined is not None and m.device != combined.device:
                m = m.to(combined.device)
            combined = m if combined is None else (combined & m)
            
        return combined

    def compute_attention(self, query, key):
        """计算注意力分数"""
        # query: (..., I, D)
        # key:   (..., J, D)
        Q = self.query_proj(query)
        K = self.key_proj(key)
        
        # scores: (..., I, J)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        return scores

    def decode(self, query_set, key_set, action_masks, top_k=32):
        """
        解码步骤：
        1. 计算 Attention Scores (如果提供了 query/key)。
        2. 结合 TTI 偏置。
        3. 应用掩码。
        4. 返回 Top-K 分配结果。
        """
        # 1. 获取合并掩码
        combined_mask = self.get_combined_mask(action_masks)
        if combined_mask is None:
            return torch.empty((0, 2), dtype=torch.long)
        
        device = query_set.device if isinstance(query_set, torch.Tensor) else torch.device('cpu')
        if isinstance(combined_mask, torch.Tensor):
            device = combined_mask.device

        # 2. 获取 TTI
        pairwise_tti = None
        if action_masks and 'pairwise_tti' in action_masks:
            pairwise_tti = action_masks['pairwise_tti']
            if not isinstance(pairwise_tti, torch.Tensor):
                pairwise_tti = torch.tensor(pairwise_tti, dtype=torch.float32, device=device)
            else:
                pairwise_tti = pairwise_tti.to(device)

        # 3. 计算 Logits
        # 这里调用 compute_logits，传入 query 和 key 以启用注意力机制
        logits = self.compute_logits(pairwise_tti, combined_mask, query=query_set, key=key_set)

        # 4. 选择 Top-K
        # logits 形状可能是 (B, T, I*J) 或 (I*J) 等，取决于输入
        # 简化起见，这里假设是在推理阶段，我们关注最后一帧或单帧
        
        # 展平 logits 以便排序
        logits_flat = logits.reshape(-1)
        
        # 过滤掉 -inf (被掩码遮蔽的部分)
        valid_indices = torch.nonzero(logits_flat > -1e8).squeeze()
        if valid_indices.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long)
            
        valid_scores = logits_flat[valid_indices]
        
        # 排序
        sorted_scores, argsort = torch.sort(valid_scores, descending=True)
        
        # 取 Top-K
        k = min(top_k, valid_indices.numel()) if top_k > 0 else valid_indices.numel()
        top_indices_flat = valid_indices[argsort[:k]]
        
        # 还原回 (i, j) 坐标
        # 需要知道 J (目标数量)
        if pairwise_tti is not None:
            J = pairwise_tti.shape[-1]
        elif combined_mask is not None:
            J = combined_mask.shape[-1]
        else:
            # Fallback
            return torch.empty((0, 2), dtype=torch.long)
            
        i_idx = top_indices_flat // J
        j_idx = top_indices_flat % J
        
        return torch.stack([i_idx, j_idx], dim=1)

    def compute_logits(self, pairwise_tti, combined_mask, state_bias=None, query=None, key=None):
        """
        计算最终的 Logits：
        Logits = Attention(Q, K) - alpha * TTI + bias + state_bias
        """
        device = self.alpha.device
        
        # 统一设备
        if pairwise_tti is not None and not isinstance(pairwise_tti, torch.Tensor):
            pairwise_tti = torch.tensor(pairwise_tti, dtype=torch.float32, device=device)
        if combined_mask is not None and not isinstance(combined_mask, torch.Tensor):
            combined_mask = torch.tensor(combined_mask, dtype=torch.bool, device=device)
            
        # 基础 Logits (偏置)
        logits = self.bias
        
        # 1. 加入 Attention 部分 (如果提供了 Query 和 Key)
        if query is not None and key is not None:
            attn_scores = self.compute_attention(query, key)
            # attn_scores: (..., I, J) 或 (..., T, T) 取决于输入
            # 如果输入是 (B, T, D)，attn_scores 是 (B, T, T)。
            # 如果这是意图（即 Self-Attention），则直接相加。
            # 如果维度不匹配 pairwise_tti (..., I, J)，则可能需要广播或调整。
            
            # 这里的处理比较 tricky，为了兼容性，如果形状完全不匹配，我们假设
            # 调用者知道他们在做什么（例如 query/key 已经是 I/J 维度）。
            # 如果不匹配，尝试广播或者忽略（但在补全任务中，我们假设它是匹配的）。
            
            logits = logits + attn_scores

        # 2. 加入 TTI 部分 (物理约束)
        if pairwise_tti is not None:
            # TTI 越小越好，所以用负号
            logits = logits - (self.alpha * pairwise_tti)
            
        # 3. 加入 State Bias (来自 Encoder 的额外标量偏置)
        if state_bias is not None:
            if not isinstance(state_bias, torch.Tensor):
                state_bias = torch.tensor(state_bias, dtype=torch.float32, device=device)
            # 广播 state_bias
            # state_bias 通常是 (B, T, 1) 或 (1, 1)
            # logits 可能是 (B, T, I, J)
            while state_bias.dim() < logits.dim():
                state_bias = state_bias.unsqueeze(-1)
            logits = logits + state_bias

        # 4. 应用掩码
        if combined_mask is not None:
            # 确保掩码形状匹配
            # combined_mask: (..., I, J)
            # logits: (..., I, J)
            neg_inf = torch.tensor(-1e9, dtype=logits.dtype, device=device)
            
            # 自动广播
            logits = torch.where(combined_mask, logits, neg_inf)
            
        return logits

