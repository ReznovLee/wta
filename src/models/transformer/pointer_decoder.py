import torch
import torch.nn as nn


class PointerDecoder(nn.Module):
    """简化版指针解码器：
    - 将编码后的表示投影为查询与键，计算匹配分数（此处为占位，主要依据掩码与 TTI 排序）。
    - 根据 action_masks 合成合法性掩码；若存在 pairwise_tti，用其进行升序排序以得到前 top_k 个分配。
    - 返回形如 [top_k, 2] 的 index 张量，或在合法对不足时返回实际数量。
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def decode(self, query_set, key_set, action_masks, top_k=32):
        # 合成合法掩码
        masks = action_masks or {}
        def get_mask(name):
            m = masks.get(name)
            if m is None:
                return None
            # 转为 torch.bool
            if isinstance(m, torch.Tensor):
                return m.bool()
            else:
                return torch.tensor(m, dtype=torch.bool)

        # 优先使用更严格的约束集合
        mask_names = ['assign_mask', 'ammo_mask', 'range_mask', 'alive_mask', 'capacity_mask', 'defense_time_mask']
        combined = None
        for mn in mask_names:
            m = get_mask(mn)
            if m is None:
                continue
            combined = m if combined is None else (combined & m)

        if combined is None:
            # 若没有任何掩码，则无法确定合法分配；返回空
            return torch.empty((0, 2), dtype=torch.long)

        # 获取合法索引列表
        idx = torch.nonzero(combined)
        if idx.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long)

        # 若提供了 pairwise_tti，则按 TTI 升序排序（更快拦截优先）
        pairwise_tti = masks.get('pairwise_tti')
        if pairwise_tti is not None:
            if not isinstance(pairwise_tti, torch.Tensor):
                pairwise_tti = torch.tensor(pairwise_tti, dtype=torch.float32)
            # 仅对合法索引的 TTI 进行排序
            tij = pairwise_tti[idx[:, 0], idx[:, 1]]
            order = torch.argsort(tij)
            idx = idx[order]

        if top_k is None or top_k <= 0:
            return idx
        return idx[:min(top_k, idx.shape[0])]

    def compute_logits(self, pairwise_tti, combined_mask, state_bias=None):
        device = self.alpha.device if hasattr(self, 'alpha') else (pairwise_tti.device if isinstance(pairwise_tti, torch.Tensor) else torch.device('cpu'))
        if not isinstance(pairwise_tti, torch.Tensor):
            pairwise_tti = torch.tensor(pairwise_tti, dtype=torch.float32, device=device)
        if not isinstance(combined_mask, torch.Tensor):
            combined_mask = torch.tensor(combined_mask, dtype=torch.bool, device=device)
        if pairwise_tti.dim() == 4:
            B, T, I, J = pairwise_tti.shape
            logits = -(self.alpha * pairwise_tti) + self.bias
            if state_bias is not None:
                if not isinstance(state_bias, torch.Tensor):
                    state_bias = torch.tensor(state_bias, dtype=torch.float32, device=device)
                logits = logits + state_bias.view(B, T, 1, 1)
            logits = logits.view(B, T, I * J)
            mask = combined_mask.view(B, T, I * J)
            neg_inf = torch.tensor(-1e9, dtype=logits.dtype, device=device)
            logits = torch.where(mask, logits, neg_inf)
            return logits
        elif pairwise_tti.dim() == 2:
            I, J = pairwise_tti.shape
            logits = -(self.alpha * pairwise_tti) + self.bias
            if state_bias is not None:
                if not isinstance(state_bias, torch.Tensor):
                    state_bias = torch.tensor(state_bias, dtype=torch.float32, device=device)
                logits = logits + state_bias.view(1, 1)
            logits = logits.view(1, 1, I * J)
            mask = combined_mask.view(1, 1, I * J)
            neg_inf = torch.tensor(-1e9, dtype=logits.dtype, device=device)
            logits = torch.where(mask, logits, neg_inf)
            return logits
        else:
            raise ValueError('pairwise_tti must be [B,T,I,J] or [I,J]')
