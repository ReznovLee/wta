from .decision_transformer import DecisionTransformer
from .implicit_q_learning import IQL


class HDTIQLPolicy:
    def __init__(self, cfg):
        self.dt = DecisionTransformer(cfg)
        self.iql = IQL(cfg)

    def pretrain_dt(self, dataset_loader=None, epochs=1):
        return {'dt_epochs': epochs}

    def improve_with_iql(self, replay_buffer=None, steps=1000):
        return {'iql_steps': steps}

    def act(self, obs, masks):
        return self.iql.act(obs, masks)

    def save(self, path):
        return

    def load(self, path):
        return