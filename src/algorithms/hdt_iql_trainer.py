from typing import Dict, Any
import os
import time
from src.utils.logger import get_logger


class HDTIQLTrainer:
    """HDT-IQL 训练器：
    - pretrain_dt: 可选离线预训练（若 DT 不可用则跳过）。
    - train_iql: 在线交互采样 + 简单更新（在无 torch 环境下自动跳过实际更新）。
    - evaluate: 运行若干评估 episodes。
    """

    def __init__(self, env, policy, logger=None, cfg=None):
        self.env = env
        self.policy = policy
        self.logger = logger or get_logger('trainer')
        self.cfg = cfg or {}

    def pretrain_dt(self, data_cfg: Dict[str, Any]):
        offline_epochs = int(self.cfg.get('training', {}).get('offline_epochs', 1))
        # 构建数据集 DataLoader（若存在离线数据集）
        if data_cfg.get('offline_dataset'):
            offline = data_cfg.get('offline_dataset', {})
            root = str(offline.get('root', 'data/trajectories'))
            max_len = int(offline.get('max_len', 256))
            # 若为相对路径，基于项目根目录解析
            project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
            if not os.path.isabs(root):
                root = os.path.join(project_root, root)
            from src.utils.offline_dataset import make_dataloader
            dl = make_dataloader(root, split='train', max_len=max_len, batch_size=self.cfg.get('training', {}).get('offline_batch_size', 8))
        else:
            dl = None
        # 调用策略的 DT 预训练（若 DT 不可用则策略内部跳过）
        return self.policy.pretrain_dt(dataset_loader=dl, epochs=offline_epochs)

    def train_iql(self, env, buffer, total_steps=1000, log_interval=100):
        cfg_log = int(self.cfg.get('training', {}).get('log_interval', log_interval))
        obs = env.reset()
        ep_ret = 0.0
        ep_len = 0
        episodes = 0
        t0 = time.time()
        last_log_t = t0
        stats = {'value_loss': 0.0, 'critic_loss': 0.0, 'actor_loss': 0.0}
        for step in range(1, int(total_steps) + 1):
            masks = env.get_action_masks()
            action = self.policy.act(obs, masks)
            next_obs, reward, done, info = env.step(action)
            buffer.add({
                'obs': obs,
                'action': action,
                'reward': float(reward),
                'next_obs': next_obs,
                'done': bool(done),
                'masks': masks,
                'info': info,
            })
            ep_ret += float(reward)
            ep_len += 1

            # 简易更新：从 buffer 采样一个 batch 调用占位 update 函数
            batch = buffer.sample(batch_size=self.cfg.get('training', {}).get('batch_size', 64))
            try:
                val_out = self.policy.iql.update_value(batch)
                q_out = self.policy.iql.update_critic(batch)
                pi_out = self.policy.iql.update_actor(batch)
                # 累计损失（占位），用于日志显示
                stats['value_loss'] += float(val_out.get('value_loss', 0.0))
                stats['critic_loss'] += float(q_out.get('critic_loss', 0.0))
                stats['actor_loss'] += float(pi_out.get('actor_loss', 0.0))
            except Exception:
                # 在无 torch 环境或未实现更新时安全跳过
                pass

            # Episode 结束处理
            if done:
                episodes += 1
                self.logger.info(f"Episode {episodes} return={ep_ret:.3f}, length={ep_len}")
                obs = env.reset()
                ep_ret = 0.0
                ep_len = 0
            else:
                obs = next_obs

            # 周期性日志
            now = time.time()
            if step % max(1, cfg_log) == 0 or (now - last_log_t) > 30:
                avg_val = stats['value_loss'] / max(1, step)
                avg_q = stats['critic_loss'] / max(1, step)
                avg_pi = stats['actor_loss'] / max(1, step)
                self.logger.info(f"Step {step}/{total_steps} | buffer={len(buffer)} | losses: value={avg_val:.4f}, critic={avg_q:.4f}, actor={avg_pi:.4f}")
                last_log_t = now

            eval_int = int(self.cfg.get('training', {}).get('eval_interval_steps', 0))
            save_int = int(self.cfg.get('training', {}).get('save_interval_steps', 0))
            if eval_int > 0 and step % eval_int == 0:
                metrics = self.evaluate(env, episodes=5)
                self.logger.info(f"Eval at step {step}: {metrics}")

            if save_int > 0 and step % save_int == 0:
                models_dir = self.cfg.get('paths', {}).get('models', 'experiments/results/models')
                root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
                out_dir = models_dir if os.path.isabs(models_dir) else os.path.join(root, models_dir)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"policy_step_{step}.pt")
                self.policy.save(out_path)
                self.logger.info(f"Saved policy checkpoint: {out_path}")


        return {'steps': total_steps, 'episodes': episodes}

    def evaluate(self, env, episodes=5, target_return=None):
        totals = []
        target_met = 0
        for ep in range(episodes):
            obs = env.reset()
            ep_ret = 0.0
            done = False
            
            # Initialize running return-to-go
            running_rtg = float(target_return) if target_return is not None else None
            
            while not done:
                masks = env.get_action_masks()
                action = self.policy.act(obs, masks, target_return=running_rtg)
                obs, reward, done, info = env.step(action)
                ep_ret += reward
                
                if running_rtg is not None:
                    running_rtg -= reward

            totals.append(ep_ret)
            if target_return is not None and ep_ret >= target_return:
                target_met += 1

            self.logger.info(f"Eval Episode {ep+1}/{episodes} return={ep_ret:.3f}")
        
        avg = sum(totals)/len(totals) if totals else 0.0
        metrics = {'avg_return': avg}
        if target_return is not None:
            metrics['target_success_rate'] = target_met / episodes
            metrics['avg_gap'] = avg - target_return
        return metrics
