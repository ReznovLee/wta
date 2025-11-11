from src.utils.logger import get_logger


class HDTIQLTrainer:
    def __init__(self, env, policy, logger=None, cfg=None):
        self.env = env
        self.policy = policy
        self.logger = logger or get_logger('trainer')
        self.cfg = cfg or {}

    def pretrain_dt(self, data_cfg):
        return self.policy.pretrain_dt(None, epochs=self.cfg.get('training', {}).get('offline_epochs', 1))

    def train_iql(self, env, buffer, total_steps=1000):
        obs = env.reset()
        for step in range(total_steps):
            masks = env.get_action_masks()
            action = self.policy.act(obs, masks)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
        return {'steps': total_steps}

    def evaluate(self, env, episodes=5):
        totals = []
        for _ in range(episodes):
            obs = env.reset()
            ep_ret = 0.0
            done = False
            while not done:
                masks = env.get_action_masks()
                action = self.policy.act(obs, masks)
                obs, reward, done, info = env.step(action)
                ep_ret += reward
            totals.append(ep_ret)
        return {'avg_return': sum(totals)/len(totals) if totals else 0.0}