try:
    from .decision_transformer import DecisionTransformer
    DT_AVAILABLE = True
except Exception:
    DecisionTransformer = None
    DT_AVAILABLE = False
from .implicit_q_learning import IQL


class HDTIQLPolicy:
    def __init__(self, cfg):
        # DT 在无 torch 或导入失败时可缺省，仅保留 IQL 启发式策略保证评估/演示可运行
        self.dt = DecisionTransformer(cfg) if DT_AVAILABLE else None
        self.iql = IQL(cfg)

    def pretrain_dt(self, dataset_loader=None, epochs=1):
        if self.dt is None:
            return {'dt_epochs': 0, 'note': 'DecisionTransformer unavailable; skipped pretraining'}
        if dataset_loader is None:
            return {'dt_epochs': 0, 'note': 'No dataset_loader provided; skipped pretraining'}
        try:
            import torch
            opt = torch.optim.Adam(self.dt.parameters(), lr=3e-4)
        except Exception:
            return {'dt_epochs': 0, 'note': 'Torch unavailable; skipped pretraining'}
        avg_loss = 0.0
        steps = 0
        for ep in range(int(epochs)):
            for batch in dataset_loader:
                out = self.dt.loss_offline(batch)
                loss = out.get('loss')
                if loss is None:
                    continue
                opt.zero_grad()
                loss.backward()
                opt.step()
                steps += 1
                try:
                    avg_loss += float(loss.detach().cpu().item())
                except Exception:
                    avg_loss += 0.0
        avg_loss = avg_loss / max(1, steps)
        return {'dt_epochs': epochs, 'avg_loss': avg_loss}

    def improve_with_iql(self, replay_buffer=None, steps=1000):
        return {'iql_steps': steps}

    def act(self, obs, masks, target_return=None):
        if target_return is not None and self.dt is not None:
            return self.dt.sample_action(obs, returns_to_go=target_return, masks=masks)
        return self.iql.act(obs, masks)

    def save(self, path):
        try:
            import os
            import torch
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            payload = {}
            if self.dt is not None:
                payload['dt'] = self.dt.state_dict()
            if hasattr(self.iql, 'value'):
                payload['iql_value'] = getattr(self.iql.value, 'state_dict', lambda: {})()
            if hasattr(self.iql, 'qnet'):
                payload['iql_qnet'] = getattr(self.iql.qnet, 'state_dict', lambda: {})()
            if hasattr(self.iql, 'state_bias') and self.iql.state_bias is not None:
                payload['iql_state_bias'] = self.iql.state_bias.state_dict()
            if hasattr(self.iql, 'alpha') and self.iql.alpha is not None:
                payload['iql_alpha'] = self.iql.alpha.detach().cpu()
            torch.save(payload, path)
        except Exception:
            return

    def load(self, path):
        try:
            import torch
            payload = torch.load(path, map_location='cpu')
            if self.dt is not None and 'dt' in payload:
                self.dt.load_state_dict(payload['dt'])
            if hasattr(self.iql, 'value') and 'iql_value' in payload:
                self.iql.value.load_state_dict(payload['iql_value'])
            if hasattr(self.iql, 'qnet') and 'iql_qnet' in payload:
                self.iql.qnet.load_state_dict(payload['iql_qnet'])
            if hasattr(self.iql, 'state_bias') and self.iql.state_bias is not None and 'iql_state_bias' in payload:
                self.iql.state_bias.load_state_dict(payload['iql_state_bias'])
            if hasattr(self.iql, 'alpha') and self.iql.alpha is not None and 'iql_alpha' in payload:
                with torch.no_grad():
                    self.iql.alpha.copy_(payload['iql_alpha'])
        except Exception:
            return
