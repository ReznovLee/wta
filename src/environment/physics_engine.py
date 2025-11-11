import numpy as np


class PhysicsEngine:
    """最小物理引擎：范围检查与简单目标推进。"""
    def __init__(self, env_cfg):
        self.env_cfg = env_cfg

    def propagate(self, targets, dt):
        for t in targets:
            if 'position' in t and 'velocity' in t:
                t['position'] = np.array(t['position']) + dt * np.array(t['velocity'])

    def in_range(self, interceptor, target):
        max_range_km = self.env_cfg.get('weapon_specs', {}).get('max_range_km', 100)
        max_range = max_range_km * 1000.0
        ip = np.array(interceptor.get('position', [0, 0, 0]))
        tp = np.array(target.get('position', [0, 0, 0]))
        dist = np.linalg.norm(ip - tp)
        return bool(dist <= max_range)

    def estimate_intercept_time(self, interceptor, target):
        ip = np.array(interceptor.get('position', [0, 0, 0]))
        tp = np.array(target.get('position', [0, 0, 0]))
        dist = np.linalg.norm(ip - tp)
        speed = float(interceptor.get('speed', 300.0))
        if speed <= 1e-6:
            return float('inf')
        return dist / speed