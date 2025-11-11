import torch
import torch.nn as nn
import torch.optim as optim


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


class IQL:
    def __init__(self, cfg, obs_dim=128, act_dim=2):
        iql_cfg = cfg.get('iql', {})
        self.discount = iql_cfg.get('discount', 0.99)
        self.expectile = iql_cfg.get('expectile', 0.7)
        self.awr_beta = iql_cfg.get('awr_beta', 1.0)
        self.value = MLP(obs_dim, 1)
        self.qnet = MLP(obs_dim + act_dim, 1)
        self.actor = MLP(obs_dim, act_dim)
        self.v_opt = optim.Adam(self.value.parameters(), lr=iql_cfg.get('critic_lr', 3e-4))
        self.q_opt = optim.Adam(self.qnet.parameters(), lr=iql_cfg.get('critic_lr', 3e-4))
        self.pi_opt = optim.Adam(self.actor.parameters(), lr=iql_cfg.get('actor_lr', 3e-4))

    def update_value(self, batch):
        loss = torch.tensor(0.0)
        return {'value_loss': loss}

    def update_critic(self, batch):
        loss = torch.tensor(0.0)
        return {'critic_loss': loss}

    def update_actor(self, batch):
        loss = torch.tensor(0.0)
        return {'actor_loss': loss}

    def act(self, obs, masks):
        am = masks['ammo_mask'] & masks['range_mask'] & masks['assign_mask']
        idx = torch.nonzero(torch.tensor(am))
        if idx.numel() == 0:
            return None
        i, j = idx[0].tolist()
        return (i, j)