class RewardCalculator:
    def __init__(self, cfg):
        self.cfg = cfg or {}
        self.alpha = float(self.cfg.get('alpha', 1.0))
        self.beta = float(self.cfg.get('beta', 1.0))
        self.delta = float(self.cfg.get('delta', 0.1))
        self.gamma = float(self.cfg.get('gamma', 0.99))
        self.zeta = float(self.cfg.get('zeta', 3.0))
        self.eta = float(self.cfg.get('eta', 1.0))
        self.unit_cost = float(self.cfg.get('unit_cost_coeff', 1.0))

    def compute_step_reward(self, assignments, outcomes, dt=1.0):
        hits_val = float(outcomes.get('hits_value_sum', 0.0))
        time_factor_sum = float(outcomes.get('time_factor_sum', 0.0))
        coop_hits = int(outcomes.get('coop_met_hits', 0))
        miss = int(outcomes.get('misses', 0))
        violation = int(outcomes.get('violations', 0))
        shots = int(outcomes.get('shots', 0))
        delay = float(outcomes.get('delay', 0.0))
        late = int(outcomes.get('late_assignments', 0))
        Rh = self.alpha * hits_val * (1.0 + time_factor_sum)
        Ccoop = 1.2 if coop_hits > 0 else 1.0
        Rh = Rh * Ccoop
        Rv = -self.beta * float(late)
        Rc = -self.delta * self.unit_cost * float(shots)
        Rdelay = -0.01 * delay
        Rviol = -2.0 * float(violation)
        Rmiss = -1.0 * float(miss)
        return float(Rh + Rv + Rc + Rdelay + Rviol + Rmiss)

    def compute_terminal_reward(self, summary):
        intercept_all = bool(summary.get('intercept_all', False))
        if intercept_all:
            return float(self.zeta)
        pv = float(summary.get('penetrated_value_sum', 0.0))
        return float(-self.eta * pv)
