import numpy as np
from .physics_engine import PhysicsEngine
from .reward_calculator import RewardCalculator
from .action_mask import build_action_masks


class WTAEnv:
    """最小可运行的 WTA 环境。"""
    def __init__(self, env_cfg, reward_cfg, model_cfg):
        self.env_cfg = env_cfg
        self.reward_calc = RewardCalculator(reward_cfg)
        self.engine = PhysicsEngine(env_cfg)
        self.t = 0
        self.horizon = env_cfg.get('horizon', 200)
        self.state = None

    def reset(self):
        num_targets = int(self.env_cfg.get('max_targets', 50))
        num_interceptors = int(self.env_cfg.get('max_interceptors', 50))
        targets = [{
            'position': np.random.uniform(-10000, 10000, size=3),
            'velocity': np.random.uniform(-50, 50, size=3),
            'value': np.random.uniform(0.5, 1.0),
            'threat': np.random.uniform(0.5, 1.0),
        } for _ in range(num_targets)]
        interceptors = [{
            'position': np.random.uniform(-10000, 10000, size=3),
            'speed': 300.0,
            'ammo': self.env_cfg.get('weapon_specs', {}).get('ammo_per_unit', 4),
            'hit_prob': 0.7,
        } for _ in range(num_interceptors)]
        self.state = {
            'targets': targets,
            'interceptors': interceptors,
            'num_targets': num_targets,
            'num_interceptors': num_interceptors,
            'interceptor_ammo': [it['ammo'] for it in interceptors],
        }
        self.t = 0
        return self.compute_observation(self.state, self.t)

    def step(self, action):
        outcomes = {'hits': 0, 'misses': 0, 'violations': 0, 'shots': 0, 'delay': 0.0}
        if action is not None:
            i, j = action
            if 0 <= i < self.state['num_interceptors'] and 0 <= j < self.state['num_targets']:
                interceptor = self.state['interceptors'][i]
                target = self.state['targets'][j]
                valid = self.engine.in_range(interceptor, target) and interceptor['ammo'] > 0
                if valid:
                    interceptor['ammo'] -= 1
                    outcomes['shots'] += 1
                    t_imp = self.engine.estimate_intercept_time(interceptor, target)
                    outcomes['delay'] += float(t_imp)
                    hit_prob = interceptor.get('hit_prob', 0.7)
                    hit = np.random.rand() < hit_prob
                    outcomes['hits'] += int(hit)
                    outcomes['misses'] += int(not hit)
                else:
                    outcomes['violations'] += 1
        self.engine.propagate(self.state['targets'], dt=self.env_cfg.get('time_step', 1.0))
        self.t += 1
        obs = self.compute_observation(self.state, self.t)
        reward = self.reward_calc.compute_step_reward(None, outcomes, dt=self.env_cfg.get('time_step', 1.0))
        done = self.t >= self.horizon
        info = {'t': self.t}
        return obs, reward, done, info

    def get_action_masks(self, state=None):
        s = state or self.state
        masks = build_action_masks(s, self.env_cfg)
        return masks

    def compute_observation(self, state, t):
        tgt = state['targets']
        itc = state['interceptors']
        obs = {
            'targets_pos': np.array([x['position'] for x in tgt]),
            'targets_val': np.array([x['value'] for x in tgt]),
            'targets_thr': np.array([x['threat'] for x in tgt]),
            'interceptors_pos': np.array([x['position'] for x in itc]),
            'interceptors_ammo': np.array([x['ammo'] for x in itc]),
            't': t,
        }
        return obs