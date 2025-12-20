import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    # 允许在未安装 PyTorch 的情况下运行评估/演示（使用启发式策略）
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None


if TORCH_AVAILABLE:
    class MLP(nn.Module):
        def __init__(self, in_dim, out_dim, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, out_dim)
            )

        def forward(self, x):
            return self.net(x)
else:
    # 占位 MLP：在无 torch 时避免导入错误，不参与计算
    class MLP:
        def __init__(self, *args, **kwargs):
            pass
        def parameters(self):
            return []
        def __call__(self, *args, **kwargs):
            return None


class IQL:
    def __init__(self, cfg, obs_dim=256, act_dim=4):
        iql_cfg = cfg.get('iql', {})
        self.discount = iql_cfg.get('discount', 0.99)
        self.expectile = iql_cfg.get('expectile', 0.7)
        self.awr_beta = iql_cfg.get('awr_beta', 1.0)
        self.epsilon = float(iql_cfg.get('epsilon', 0.05))
        self.temperature = float(iql_cfg.get('temperature', 1.0))
        self.device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() and str(cfg.get('device', 'auto')) != 'cpu' else 'cpu')
        self.value = MLP(obs_dim, 1)
        self.qnet = MLP(obs_dim + act_dim, 1)
        if TORCH_AVAILABLE:
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.state_bias = nn.Linear(obs_dim, 1)
        else:
            self.alpha = None
            self.state_bias = None
        if TORCH_AVAILABLE:
            self.v_opt = optim.Adam(self.value.parameters(), lr=iql_cfg.get('critic_lr', 3e-4))
            self.q_opt = optim.Adam(self.qnet.parameters(), lr=iql_cfg.get('critic_lr', 3e-4))
            self.pi_opt = optim.Adam(list(self.state_bias.parameters()) + [self.alpha], lr=iql_cfg.get('actor_lr', 3e-4))
        else:
            self.v_opt = None
            self.q_opt = None
            self.pi_opt = None
        if TORCH_AVAILABLE:
            self.value.to(self.device)
            self.qnet.to(self.device)
            self.state_bias.to(self.device)

    def update_value(self, batch):
        if not TORCH_AVAILABLE:
            return {'value_loss': 0.0}
        s, a_idx, a_feat, r, d, s2 = self._prepare_batch(batch)
        q = self.qnet(torch.cat([s, a_feat], dim=1)).detach()
        v = self.value(s)
        err = q - v
        w = torch.abs(torch.where(err < 0, torch.tensor(1.0 - self.expectile, device=err.device), torch.tensor(self.expectile, device=err.device)))
        loss = (w * err.pow(2)).mean()
        self.v_opt.zero_grad()
        loss.backward()
        self.v_opt.step()
        return {'value_loss': float(loss.detach().cpu().item())}

    def update_critic(self, batch):
        if not TORCH_AVAILABLE:
            return {'critic_loss': 0.0}
        s, a_idx, a_feat, r, d, s2 = self._prepare_batch(batch)
        with torch.no_grad():
            v2 = self.value(s2)
            y = r + (1.0 - d) * self.discount * v2
        q = self.qnet(torch.cat([s, a_feat], dim=1))
        loss = torch.mean((q - y).pow(2))
        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()
        return {'critic_loss': float(loss.detach().cpu().item())}

    def update_actor(self, batch):
        if not TORCH_AVAILABLE:
            return {'actor_loss': 0.0}
        s, a_idx, a_feat, r, d, s2, pairwise_tti_list, mask_list, IJ_list = self._prepare_actor_batch(batch)
        with torch.no_grad():
            q = self.qnet(torch.cat([s, a_feat], dim=1))
            v = self.value(s)
            adv = q - v
            w = torch.exp(self.awr_beta * adv).clamp(max=100.0)
        losses = []
        self.pi_opt.zero_grad()
        for k in range(len(pairwise_tti_list)):
            tti_mat = pairwise_tti_list[k]
            mask = mask_list[k]
            I, J = IJ_list[k]
            sb = self.state_bias(s[k:k+1]).view(1, 1)
            logits = -(self.alpha * tti_mat) + sb
            logits = logits.view(1, I * J)
            mask_flat = mask.view(1, I * J)
            neg_inf = torch.full_like(logits, -1e9)
            logits = torch.where(mask_flat, logits, neg_inf)
            if self.temperature is not None and self.temperature > 1e-6:
                logits = logits / self.temperature
            target = a_idx[k:k+1]
            ce = torch.nn.functional.cross_entropy(logits, target)
            losses.append(w[k] * ce)
        loss = torch.stack(losses).mean()
        loss.backward()
        self.pi_opt.step()
        return {'actor_loss': float(loss.detach().cpu().item())}

    def act(self, obs, masks):
        """策略执行：
        - 合并硬约束掩码（弹药、射程、存活、容量、防御时间窗、基础分配掩码）；
        - 若存在 pairwise_tti：
            * 在有 torch 的情况下，使用学习到的 logits：logits_ij = -(alpha)*TTI_ij + state_bias(s)，在合法集合上取 argmax；
            * 在无 torch 的情况下，选择 TTI 最小的合法对；
          否则选择第一个合法对。
        - 返回 list[(i, j)]。
        """
        assign = masks.get('assign_mask')
        alive = masks.get('alive_mask')
        range_mask = masks.get('range_mask')
        capacity = masks.get('capacity_mask')
        defense = masks.get('defense_time_mask')
        ammo = masks.get('ammo_mask')
        # 容错：若某些掩码缺失，则用全 True 矩阵替代
        def ones_like(x):
            return np.ones_like(x, dtype=bool) if x is not None else None
        base = assign if assign is not None else (range_mask or alive or capacity or defense or ammo)
        if base is None:
            return []
        if alive is None: alive = ones_like(base)
        if range_mask is None: range_mask = ones_like(base)
        if capacity is None: capacity = ones_like(base)
        if defense is None: defense = ones_like(base)
        if ammo is None: ammo = ones_like(base)
        if assign is None: assign = ones_like(base)

        combined = assign & alive & range_mask & capacity & defense & ammo
        idxs = np.argwhere(combined)
        if idxs.size == 0:
            return []

        tti = masks.get('pairwise_tti')
        if isinstance(tti, np.ndarray) and tti.shape == combined.shape:
            if TORCH_AVAILABLE and self.alpha is not None and self.state_bias is not None:
                s_feat = self._obs_to_feature(obs)
                sb = self.state_bias(s_feat.view(1, -1)).view(1, 1)
                tti_t = torch.tensor(tti, dtype=torch.float32)
                mask_t = torch.tensor(combined, dtype=torch.bool)
                I, J = int(tti_t.shape[0]), int(tti_t.shape[1])
                logits = -(self.alpha * tti_t) + sb
                logits = logits.view(1, I * J)
                mask_flat = mask_t.view(1, I * J)
                neg_inf = torch.full_like(logits, -1e9)
                logits = torch.where(mask_flat, logits, neg_inf)
                if np.random.rand() < self.epsilon:
                    legal = np.where(mask_flat.view(-1).cpu().numpy())[0]
                    if legal.size > 0:
                        pick = int(np.random.choice(legal))
                        i, j = int(pick // J), int(pick % J)
                        return [(int(i), int(j))]
                if self.temperature is not None and self.temperature > 1e-6:
                    probs = torch.nn.functional.softmax(logits.view(-1) / self.temperature, dim=0)
                    idx = torch.multinomial(probs, num_samples=1).item()
                else:
                    idx = int(torch.argmax(logits.view(-1)).item())
                i, j = int(idx // J), int(idx % J)
                return [(int(i), int(j))]
            else:
                idxs_sorted = sorted(idxs.tolist(), key=lambda ij: tti[ij[0], ij[1]])
                i, j = idxs_sorted[0]
                return [(int(i), int(j))]
        else:
            i, j = idxs[0]
            return [(int(i), int(j))]

    def _obs_to_feature(self, obs):
        if not TORCH_AVAILABLE:
            return None
        tp = obs.get('targets_pos')
        ip = obs.get('interceptors_pos')
        alive = obs.get('targets_alive')
        tti = obs.get('pairwise_tti')
        dist = obs.get('pairwise_distance')
        val = obs.get('targets_val')
        thr = obs.get('targets_thr')
        ammo_arr = obs.get('interceptors_ammo')
        hp_arr = obs.get('interceptors_hit_prob')
        spd_arr = obs.get('interceptors_speed')
        type_id = obs.get('targets_type_id')
        tcur = obs.get('t', 0.0)
        nt = int(len(tp) if tp is not None else 0)
        ni = int(len(ip) if ip is not None else 0)
        def _stats_np(a):
            if a is None:
                return 0.0, 0.0, 0.0
            arr = np.asarray(a, dtype=float)
            if arr.size == 0:
                return 0.0, 0.0, 0.0
            return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr))
        ammo_mean, ammo_min, ammo_max = _stats_np(ammo_arr)
        hp_mean, hp_min, hp_max = _stats_np(hp_arr)
        spd_mean, spd_min, spd_max = _stats_np(spd_arr)
        val_mean, val_min, val_max = _stats_np(val)
        thr_mean, thr_min, thr_max = _stats_np(thr)
        tti_mean, tti_min, tti_max = _stats_np(tti)
        dist_mean, dist_min, dist_max = _stats_np(dist)
        alive_sum = float(sum(alive) if isinstance(alive, (list, np.ndarray)) and len(alive) > 0 else 0.0)
        type_hist = np.zeros(3, dtype=float)
        if isinstance(type_id, (list, np.ndarray)) and len(type_id) > 0:
            arr = np.asarray(type_id, dtype=int)
            for k in [0, 1, 2]:
                type_hist[k] = float(np.sum(arr == k))
            if nt > 0:
                type_hist = type_hist / float(nt)
        feat = [
            float(nt),
            float(ni),
            alive_sum,
            ammo_mean,
            ammo_min,
            ammo_max,
            hp_mean,
            hp_min,
            hp_max,
            spd_mean,
            spd_min,
            spd_max,
            val_mean,
            val_min,
            val_max,
            thr_mean,
            thr_min,
            thr_max,
            float(tcur),
            tti_mean,
            tti_min,
            tti_max,
            dist_mean,
            dist_min,
            dist_max,
        ] + list(type_hist.astype(float))
        if len(feat) < 256:
            feat = feat + [0.0] * (256 - len(feat))
        else:
            feat = feat[:256]
        return torch.tensor(feat, dtype=torch.float32, device=self.device)

    def _prepare_batch(self, batch):
        s_list = []
        a_idx_list = []
        a_feat_list = []
        r_list = []
        d_list = []
        s2_list = []
        for tr in batch:
            obs = tr.get('obs', {})
            next_obs = tr.get('next_obs', {})
            masks = tr.get('masks', {})
            action = tr.get('action', [])
            r = float(tr.get('reward', 0.0))
            done = float(bool(tr.get('done', False)))
            s_feat = self._obs_to_feature(obs)
            s2_feat = self._obs_to_feature(next_obs)
            if s_feat is None or s2_feat is None:
                continue
            assign = masks.get('assign_mask')
            if action and isinstance(action, list) and len(action) > 0 and assign is not None:
                i, j = action[0]
                J = len(assign[0]) if len(assign) > 0 else 0
                a_idx = int(i) * int(J) + int(j)
                tti = masks.get('pairwise_tti')
                tij = float(tti[i][j]) if isinstance(tti, np.ndarray) else float(tti[i][j])
                dist = obs.get('pairwise_distance')
                dij = float(dist[i][j]) if dist is not None else 0.0
                hp = obs.get('interceptors_hit_prob')
                hpi = float(hp[i]) if hp is not None and len(hp) > i else 0.0
                spd = obs.get('interceptors_speed')
                si = float(spd[i]) if spd is not None and len(spd) > i else 0.0
                a_feat = [tij, dij, hpi, si]
            else:
                a_idx = 0
                a_feat = [0.0, 0.0, 0.0, 0.0]
            s_list.append(s_feat)
            a_idx_list.append(a_idx)
            a_feat_list.append(a_feat)
            r_list.append(r)
            d_list.append(done)
            s2_list.append(s2_feat)
        if len(s_list) == 0:
            s = torch.zeros((1, 256), dtype=torch.float32, device=self.device)
            a_idx = torch.zeros((1,), dtype=torch.long, device=self.device)
            a_feat = torch.zeros((1, 4), dtype=torch.float32, device=self.device)
            r = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
            d = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
            s2 = torch.zeros((1, 256), dtype=torch.float32, device=self.device)
            return s, a_idx, a_feat, r, d, s2
        s = torch.stack(s_list, dim=0).to(self.device)
        a_idx = torch.tensor(a_idx_list, dtype=torch.long, device=self.device)
        a_feat = torch.tensor(a_feat_list, dtype=torch.float32, device=self.device)
        r = torch.tensor(r_list, dtype=torch.float32, device=self.device).view(-1, 1)
        d = torch.tensor(d_list, dtype=torch.float32, device=self.device).view(-1, 1)
        s2 = torch.stack(s2_list, dim=0).to(self.device)
        return s, a_idx, a_feat, r, d, s2

    def _prepare_actor_batch(self, batch):
        s, a_idx, a_feat, r, d, s2 = self._prepare_batch(batch)
        pairwise_tti_list = []
        mask_list = []
        IJ_list = []
        for tr in batch:
            masks = tr.get('masks', {})
            tti = masks.get('pairwise_tti')
            assign = masks.get('assign_mask')
            alive = masks.get('alive_mask')
            range_m = masks.get('range_mask')
            capacity = masks.get('capacity_mask')
            defense = masks.get('defense_time_mask')
            ammo = masks.get('ammo_mask')
            combined = None
            for m in [assign, alive, range_m, capacity, defense, ammo]:
                if m is None:
                    continue
                combined = m if combined is None else (combined & m)
            if isinstance(tti, np.ndarray):
                tti_t = torch.tensor(tti, dtype=torch.float32, device=self.device)
            else:
                tti_t = torch.tensor(tti, dtype=torch.float32, device=self.device)
            mask_t = torch.tensor(combined, dtype=torch.bool, device=self.device) if combined is not None else torch.zeros_like(tti_t, dtype=torch.bool)
            I, J = int(tti_t.shape[0]), int(tti_t.shape[1])
            pairwise_tti_list.append(tti_t)
            mask_list.append(mask_t)
            IJ_list.append((I, J))
        return s, a_idx, a_feat, r, d, s2, pairwise_tti_list, mask_list, IJ_list
