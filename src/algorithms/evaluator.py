def run_eval(policy, env, episodes=10, metrics_fn=None):
    totals = []
    for _ in range(episodes):
        obs = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            masks = env.get_action_masks()
            action = policy.act(obs, masks)
            obs, reward, done, info = env.step(action)
            ep_ret += reward
        totals.append(ep_ret)
    return {'avg_return': sum(totals)/len(totals) if totals else 0.0}