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

    def forward(self, states, returns_to_go, action_masks):
        if returns_to_go is not None:
            if not isinstance(returns_to_go, torch.Tensor):
                returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32)
            if returns_to_go.dim() == 2:
                returns_to_go = returns_to_go.unsqueeze(-1)
            rtg_feat = self.rtg_proj(returns_to_go)
            states = states + rtg_feat
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        states = states.to(self.device)
        if self.use_hier:
            B, T, D = states.shape
            half = max(1, T//2)
            targets_seq = states[:, :half, :]
            interceptors_seq = states[:, half:, :]
            global_state = torch.mean(states, dim=1)
            enc = self.encoder_hier(targets_seq, interceptors_seq, global_state)
        else:
            enc = self.encoder(states)
        assignments = self.decoder.decode(enc, enc, action_masks, top_k=self.top_k)
        return assignments

    def loss_offline(self, batch):
        states = batch.get('states')
        action_masks = batch.get('action_masks')
        pairwise_tti = batch.get('pairwise_tti')
        returns_to_go = batch.get('returns_to_go')
        actions = batch.get('actions', [])
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        if not isinstance(action_masks, torch.Tensor):
            action_masks = torch.tensor(action_masks, dtype=torch.bool)
        if not isinstance(pairwise_tti, torch.Tensor):
            pairwise_tti = torch.tensor(pairwise_tti, dtype=torch.float32)
        if returns_to_go is not None and not isinstance(returns_to_go, torch.Tensor):
            returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32)
        B, T, I, J = action_masks.shape
        if returns_to_go is not None:
            if returns_to_go.dim() == 2:
                returns_to_go = returns_to_go.unsqueeze(-1)
            rtg_feat = self.rtg_proj(returns_to_go)
            states = states + rtg_feat
        states = states.to(self.device)
        if self.use_hier:
            B, T, D = states.shape
            half = max(1, T//2)
            targets_seq = states[:, :half, :]
            interceptors_seq = states[:, half:, :]
            global_state = torch.mean(states, dim=1)
            enc = self.encoder_hier(targets_seq, interceptors_seq, global_state)
        else:
            enc = self.encoder(states)
        combined = action_masks
        sb = self.state_bias(enc)
        logits = self.decoder.compute_logits(pairwise_tti, combined, state_bias=sb)
        targets = []
        for b in range(B):
            for t in range(T):
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

    def sample_action(self, obs, masks):
        # 合成掩码（包含更严格的约束）
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

        idx = torch.nonzero(combined)
        if idx.numel() == 0:
            return None

        pairwise_tti = masks.get('pairwise_tti')
        if pairwise_tti is not None:
            tti = torch.tensor(pairwise_tti) if not isinstance(pairwise_tti, torch.Tensor) else pairwise_tti
            logits = self.decoder.compute_logits(tti, combined, state_bias=None)
            flat = logits.view(-1)
            if self.temperature is not None and self.temperature > 1e-6:
                probs = torch.nn.functional.softmax(flat / self.temperature, dim=0)
                idx = torch.multinomial(probs, num_samples=1).item()
            else:
                idx = int(torch.argmax(flat).item())
            I, J = combined.shape
            i = int(idx // J)
            j = int(idx % J)
        else:
            i, j = idx[0].tolist()
        return [(i, j)]
