class RewardCalculator:
    """基于配置权重的确定性奖励计算器。"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.w = cfg.get('weights', {})
        self.shaping = cfg.get('shaping', {})
        self.terminal = cfg.get('terminal_bonus', {})

    def compute_step_reward(self, assignments, outcomes, dt=1.0):
        hit = outcomes.get('hits', 0)
        miss = outcomes.get('misses', 0)
        violation = outcomes.get('violations', 0)
        shots = outcomes.get('shots', 0)
        delay = outcomes.get('delay', 0.0)
        r = 0.0
        r += self.w.get('hit', 1.0) * hit
        r += self.w.get('miss', -1.0) * miss
        r += self.w.get('violation', -2.0) * violation
        r += self.w.get('resource_cost_per_shot', -0.1) * shots
        r += self.w.get('delay_penalty_per_s', -0.01) * delay
        return float(r)

    def compute_terminal_reward(self, summary):
        intercept_all = summary.get('intercept_all', False)
        return float(self.terminal.get('intercept_all', 0.0) if intercept_all else 0.0)