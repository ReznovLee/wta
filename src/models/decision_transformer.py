import torch
import torch.nn as nn
from .transformer.spatial_temporal_encoder import SpatialTemporalEncoder
from .transformer.hierarchical_encoder import HierarchicalEncoder
from .transformer.pointer_decoder import PointerDecoder


class DecisionTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dt_cfg = cfg.get('decision_transformer', {})
        d_model = dt_cfg.get('d_model', 256)
        n_heads = dt_cfg.get('n_heads', 8)
        n_layers = dt_cfg.get('n_layers', 4)
        dropout = dt_cfg.get('dropout', 0.1)
        self.device = torch.device('cuda' if torch.cuda.is_available() and str(cfg.get('device', 'auto')) != 'cpu' else 'cpu')
        use_hier = bool(cfg.get('encoder', {}).get('hierarchical_attention', False))
        self.use_hier = use_hier
        if use_hier:
            self.encoder_hier = HierarchicalEncoder(d_model=d_model, n_heads=n_heads, n_layers=max(1, n_layers//2), dropout=dropout)
            self.encoder = None
        else:
            self.encoder = SpatialTemporalEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.decoder = PointerDecoder(d_model=d_model)
        self.top_k = dt_cfg.get('pointer_top_k', 32)
        self.temperature = float(dt_cfg.get('temperature', 1.0))
        self.state_bias = nn.Linear(d_model, 1)
        self.rtg_proj = nn.Linear(1, d_model)
        self.criterion = nn.CrossEntropyLoss()
        for mod in [self.encoder_hier if use_hier else self.encoder, self.decoder, self.state_bias, self.rtg_proj]:
            if mod is not None:
                mod.to(self.device)

    def make_block_mask(self, T, N_row, N_col=None):
        """生成 Block Causal Mask:
        - 维度: (T*N_row, T*N_col)
        - 规则: 
            1. 时刻 t 的实体可以看到时刻 0...t 的所有实体 (Time Causal)
            2. 时刻 t 的实体不能看到时刻 t+1...T 的实体 (Future Masking)
            3. 时刻 t 内部的所有实体可以互相看到 (Fully Connected within Step)
        """
        if N_col is None:
            N_col = N_row
            
        # Row indices (Query)
        frames_row = torch.arange(T, device=self.device).unsqueeze(1).expand(T, N_row).reshape(-1).unsqueeze(1) # (T*N_row, 1)
        
        # Col indices (Key)
        frames_col = torch.arange(T, device=self.device).unsqueeze(1).expand(T, N_col).reshape(-1).unsqueeze(0) # (1, T*N_col)
        
        # Mask condition: col_time <= row_time
        mask_bool = frames_col <= frames_row
        
        mask = torch.zeros(T*N_row, T*N_col, device=self.device)
        mask.masked_fill_(~mask_bool, float('-inf'))
        return mask

    def _encode(self, states, returns_to_go=None, padding_mask=None):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        
        # 统一处理为序列格式 (B, T, N, D)
        if states.dim() == 3: # (B, N, D)
             states = states.unsqueeze(1) # (B, 1, N, D)
        
        B, T, N_total, D = states.shape
        
        if returns_to_go is not None:
            if not isinstance(returns_to_go, torch.Tensor):
                returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32)
            if returns_to_go.dim() == 2:
                returns_to_go = returns_to_go.unsqueeze(-1)
            # rtg: (B, T, 1) -> (B, T, 1, D)
            rtg_feat = self.rtg_proj(returns_to_go).unsqueeze(2)
            states = states + rtg_feat
            
        states = states.to(self.device)
        
        # 处理 Padding Mask
        t_pm = None
        i_pm = None
        flat_pm = None
        
        if padding_mask is not None:
            # 假设 padding_mask 为 (B, T)，True 表示 Padding (Ignored)
            if not isinstance(padding_mask, torch.Tensor):
                padding_mask = torch.tensor(padding_mask, device=self.device)
            else:
                padding_mask = padding_mask.to(self.device)
            
            if padding_mask.dim() == 2:
                if self.use_hier:
                    half = max(1, N_total//2)
                    # Expand (B, T) -> (B, T, Nt) -> (B, T*Nt)
                    t_pm = padding_mask.unsqueeze(2).expand(B, T, half).reshape(B, T*half)
                    i_pm = padding_mask.unsqueeze(2).expand(B, T, N_total-half).reshape(B, T*(N_total-half))
                else:
                    flat_pm = padding_mask.unsqueeze(2).expand(B, T, N_total).reshape(B, T*N_total)
        
        if self.use_hier:
            half = max(1, N_total//2)
            targets_seq = states[:, :, :half, :] 
            interceptors_seq = states[:, :, half:, :]
            
            B, T, Nt, D = targets_seq.shape
            _, _, Ni, _ = interceptors_seq.shape
            
            targets_flat = targets_seq.reshape(B, T * Nt, D)
            interceptors_flat = interceptors_seq.reshape(B, T * Ni, D)
            
            t_mask = self.make_block_mask(T, Nt)
            i_mask = self.make_block_mask(T, Ni)
            cross_mask = self.make_block_mask(T, Ni, Nt)
            
            global_state = torch.mean(states, dim=2) 
            g_expanded = global_state.repeat_interleave(Ni, dim=1)
            
            enc, t_enc, i_enc = self.encoder_hier(
                targets_flat, 
                interceptors_flat, 
                g_expanded,
                t_mask=t_mask,
                i_mask=i_mask,
                cross_mask=cross_mask,
                t_padding_mask=t_pm,
                i_padding_mask=i_pm
            )
            
            t_enc = t_enc.view(B, T, Nt, D)
            i_enc = i_enc.view(B, T, Ni, D)
            
        else:
            B, T, N, D = states.shape
            states_flat = states.reshape(B, T * N, D)
            mask = self.make_block_mask(T, N)
            enc = self.encoder(states_flat, mask=mask, src_key_padding_mask=flat_pm)
            enc = enc.view(B, T, N, D)
            half = max(1, N // 2)
            t_enc = enc[:, :, :half, :]
            i_enc = enc[:, :, half:, :]

        # 计算 State Bias
        state_bias = self.state_bias(i_enc)
        
        return i_enc, t_enc, state_bias

    def forward(self, states, returns_to_go, action_masks, padding_mask=None):
        # 注意：forward 的输入 states 通常是单帧或序列。
        # 如果是序列，shape 为 (B, T, N, D)
        # 如果是单帧，shape 为 (B, N, D)
        
        i_enc, t_enc, _ = self._encode(states, returns_to_go, padding_mask)
        
        # Decoder 需要处理 Batch 和 Time。
        # 我们的 Decoder 实现似乎没有显式处理 Time 维度，但它是 Pointwise/Parallel 的。
        # compute_logits 接收 (..., I, D) 和 (..., J, D) 并广播。
        # action_masks: (B, T, Ni, Nt)
        
        assignments = self.decoder.decode(i_enc, t_enc, action_masks, top_k=self.top_k)
        return assignments

    def loss_offline(self, batch):
        states = batch.get('states')
        action_masks = batch.get('action_masks')
        pairwise_tti = batch.get('pairwise_tti')
        returns_to_go = batch.get('returns_to_go')
        attention_mask = batch.get('attention_mask')
        actions = batch.get('actions', [])
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        if not isinstance(action_masks, torch.Tensor):
            action_masks = torch.tensor(action_masks, dtype=torch.bool)
        if not isinstance(pairwise_tti, torch.Tensor):
            pairwise_tti = torch.tensor(pairwise_tti, dtype=torch.float32)
        if returns_to_go is not None and not isinstance(returns_to_go, torch.Tensor):
            returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32)
        
        # 确保 states 是 (B, T, N, D)
        if states.dim() == 3: # (B*T, N, D) ? 还是 (B, N, D)?
             # 假设 batch 中已经包含了 B 和 T 维度
             pass
             
        # 获取维度
        B, T, I, J = action_masks.shape
        # 如果 states 已经被 Flatten 成了 (B*T, N, D)，我们需要恢复它
        if states.dim() == 3 and states.shape[0] == B*T:
            states = states.view(B, T, -1, states.shape[-1])
            
        states = states.to(self.device)
        action_masks = action_masks.to(self.device)
        pairwise_tti = pairwise_tti.to(self.device)
        
        padding_mask = None
        if attention_mask is not None:
             if not isinstance(attention_mask, torch.Tensor):
                 attention_mask = torch.tensor(attention_mask, device=self.device)
             else:
                 attention_mask = attention_mask.to(self.device)
             padding_mask = (attention_mask == 0)
        
        # 使用 _encode 进行编码
        i_enc, t_enc, sb = self._encode(states, returns_to_go, padding_mask)
        
        # 计算 Logits
        # query (i_enc): (B, T, I, D)
        # key (t_enc): (B, T, J, D)
        combined = action_masks
        
        logits = self.decoder.compute_logits(pairwise_tti, combined, state_bias=sb, query=i_enc, key=t_enc)
        
        targets = []

        # ... (targets 处理保持不变)
        B = B_mask # 恢复 B 变量名含义以匹配后续代码
        T = T_mask
                a = actions[b][t] if b < len(actions) and t < len(actions[b]) else []
                if a and isinstance(a, list) and len(a) > 0:
                    i, j = a[0]
                    targets.append(int(i) * J + int(j))
                else:
                    targets.append(0)
        targets = torch.tensor(targets, dtype=torch.long, device=logits.device)
        logits_flat = logits.view(B * T, I * J)
        if returns_to_go is not None:
            rtg_flat = returns_to_go.view(B * T)
            w = rtg_flat / (rtg_flat.mean() + 1e-6)
        else:
            w = torch.ones(B * T, dtype=logits_flat.dtype, device=logits_flat.device)
        log_probs = torch.nn.functional.log_softmax(logits_flat, dim=1)
        nll = -log_probs[torch.arange(B * T, device=logits_flat.device), targets]
        loss = (w * nll).mean()
        return {'loss': loss}

    def sample_action(self, obs, masks, returns_to_go=None):
        # 1. 提取掩码
        def get_mask(name):
            m = masks.get(name)
            if m is None:
                return None
            return torch.tensor(m) if not isinstance(m, torch.Tensor) else m

        combined = None
        for mn in ['assign_mask', 'ammo_mask', 'range_mask', 'alive_mask', 'capacity_mask', 'defense_time_mask']:
            m = get_mask(mn)
            if m is None:
                continue
            m = m.bool()
            combined = m if combined is None else (combined & m)

        if combined is None:
            return None
        
        # 确保 combined 是 Tensor
        if not isinstance(combined, torch.Tensor):
             combined = torch.tensor(combined, dtype=torch.bool, device=self.device)
        else:
             combined = combined.to(self.device)

        idx = torch.nonzero(combined)
        if idx.numel() == 0:
            return None

        # 2. 调用 Encoder
        # obs 应该是 (N, D) 或 (1, N, D)，需要转为 tensor
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
        else:
            obs_tensor = obs
        
        # 如果是 (N, D)，增加 Batch 和 Time 维度 -> (1, 1, N, D)
        if obs_tensor.dim() == 2:
            obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)
        elif obs_tensor.dim() == 3: # (B, N, D)
            obs_tensor = obs_tensor.unsqueeze(1)
            
        # Returns to go
        rtg = None
        if returns_to_go is not None:
             # 支持标量或 tensor
             if not isinstance(returns_to_go, torch.Tensor):
                 rtg = torch.tensor(returns_to_go, dtype=torch.float32).view(1, 1, 1)
             elif returns_to_go.dim() == 0:
                 rtg = returns_to_go.view(1, 1, 1)
             elif returns_to_go.dim() == 1:
                 rtg = returns_to_go.view(1, 1, 1)
             else:
                 rtg = returns_to_go
             
        i_enc, t_enc, state_bias = self._encode(obs_tensor, rtg)
        
        # i_enc: (1, 1, I, D), t_enc: (1, 1, J, D), state_bias: (1, 1, I, 1)
        
        pairwise_tti = masks.get('pairwise_tti')
        if pairwise_tti is not None:
            tti = torch.tensor(pairwise_tti, device=self.device) if not isinstance(pairwise_tti, torch.Tensor) else pairwise_tti.to(self.device)
            # 确保 tti 维度匹配
            if tti.dim() == 2:
                tti = tti.unsqueeze(0).unsqueeze(0) # (1, 1, I, J)
            
            # combined 需要 (1, 1, I, J)
            combined_input = combined
            if combined_input.dim() == 2:
                combined_input = combined_input.unsqueeze(0).unsqueeze(0)

            logits = self.decoder.compute_logits(tti, combined_input, state_bias=state_bias, query=i_enc, key=t_enc)
            # logits: (1, 1, I, J)
            
            flat = logits.view(-1)
            
            if self.temperature is not None and self.temperature > 1e-6:
                probs = torch.nn.functional.softmax(flat / self.temperature, dim=0)
                idx = torch.multinomial(probs, num_samples=1).item()
            else:
                idx = int(torch.argmax(flat).item())
            
            I, J = combined.shape[-2], combined.shape[-1]
            i = int(idx // J)
            j = int(idx % J)
            
            # 校验是否有效 (虽然 softmax 会抑制 -inf，但为了保险)
            if not combined.view(-1)[idx]:
                 # Fallback
                 valid_indices = torch.nonzero(combined.view(-1)).squeeze()
                 if valid_indices.numel() > 0:
                     if valid_indices.dim() == 0: # single element
                         idx = valid_indices.item()
                     else:
                         idx = valid_indices[torch.randint(0, valid_indices.numel(), (1,)).item()]
                     i = int(idx // J)
                     j = int(idx % J)
                 else:
                     return []

        else:
            # TTI 缺失时的回退
            if idx.numel() == 0:
                return []
            else:
                i, j = idx[0].tolist()
                
        return [(i, j)]
