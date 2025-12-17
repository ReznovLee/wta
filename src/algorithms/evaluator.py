def run_eval(policy, env, episodes=10, metrics_fn=None):
    totals = []
    hits = []
    misses = []
    violations = []
    late_assignments = []
    capacity_violations = []
    delays = []
    for _ in range(episodes):
        obs = env.reset()
        ep_ret = 0.0
        done = False
        ep_hits = 0
        ep_miss = 0
        ep_violate = 0
        ep_late = 0
        ep_cap = 0
        ep_delay = 0.0
        while not done:
            masks = env.get_action_masks()
            action = policy.act(obs, masks)
            obs, reward, done, info = env.step(action)
            ep_ret += reward
            ep_hits += int(info.get('hits', 0))
            ep_miss += int(info.get('misses', 0))
            ep_violate += int(info.get('violations', 0))
            ep_late += int(info.get('late_assignments', 0))
            ep_cap += int(info.get('capacity_violations', 0))
            ep_delay += float(info.get('delay', 0.0))
        totals.append(ep_ret)
        hits.append(ep_hits)
        misses.append(ep_miss)
        violations.append(ep_violate)
        late_assignments.append(ep_late)
        capacity_violations.append(ep_cap)
        delays.append(ep_delay)
    out = {
        'avg_return': sum(totals)/len(totals) if totals else 0.0,
        'avg_hits': sum(hits)/len(hits) if hits else 0.0,
        'avg_misses': sum(misses)/len(misses) if misses else 0.0,
        'avg_violations': sum(violations)/len(violations) if violations else 0.0,
        'avg_late_assignments': sum(late_assignments)/len(late_assignments) if late_assignments else 0.0,
        'avg_capacity_violations': sum(capacity_violations)/len(capacity_violations) if capacity_violations else 0.0,
        'avg_delay_proxy': sum(delays)/len(delays) if delays else 0.0,
    }
    if callable(metrics_fn):
        extra = metrics_fn()
        if isinstance(extra, dict):
            out.update(extra)
    return out
