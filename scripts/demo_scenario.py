#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Demo: 使用场景生成器初始化 WTA 环境并运行若干步

功能目标：
- 构造 env_cfg（或从 config/env.yaml 载入并补充关键参数）；
- 演示 WTAEnv.reset 现在会优先调用 scenario_generator 生成更完整的目标/拦截器；
- 每步根据掩码选择合法分配，运行环境并打印关键信息（TTI、距离、命中统计）。

运行：
  python scripts/demo_scenario.py --steps 20

可选：你也可以通过 make_scenario 先生成场景，再传入 WTAEnv。
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
try:
    import yaml
except Exception:
    yaml = None

# 保障可以从脚本目录运行（将项目根目录加入 sys.path）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.wta_env import WTAEnv


def load_env_cfg(path: str):
    cfg = {}
    if yaml is not None and os.path.exists(path):
        try:
            with open(path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
    # 补充关键配置，确保与 physics_engine/scenario_generator 对齐
    cfg.setdefault('seed', 42)
    # 为提升交互式调试效率，调小默认规模与时间
    cfg.setdefault('horizon', 120)
    cfg.setdefault('time_step', 1.0)
    cfg.setdefault('max_targets', 12)
    cfg.setdefault('max_interceptors', 6)

    cfg.setdefault('weapon_specs', {})
    cfg['weapon_specs'].setdefault('max_range_km', 100)
    cfg['weapon_specs'].setdefault('ammo_per_unit', 4)
    cfg['weapon_specs'].setdefault('speed_mps', 300.0)
    cfg['weapon_specs'].setdefault('base_hit_prob', 0.7)

    cfg.setdefault('defended_zone', {})
    cfg['defended_zone'].setdefault('center', [0.0, 0.0, 0.0])
    cfg['defended_zone'].setdefault('radius_m', 5000.0)
    cfg['defended_zone'].setdefault('center_mode', 'fixed')

    cfg.setdefault('motion_specs', {})
    ms = cfg['motion_specs']
    ms.setdefault('ballistic', {})
    ms['ballistic'].setdefault('gravity_mps2', 9.81)
    ms['ballistic'].setdefault('scatter_sigma_m', 0.0)
    ms['ballistic'].setdefault('speed_jitter_std', 0.0)

    ms.setdefault('cruise', {})
    ms['cruise'].setdefault('lateral_amp_m', 20.0)
    ms['cruise'].setdefault('lateral_omega', 0.01)
    ms['cruise'].setdefault('dive_threshold_m', 3000.0)
    ms['cruise'].setdefault('dive_rate_mps', 100.0)
    ms['cruise'].setdefault('speed_jitter_std', 0.0)
    ms['cruise'].setdefault('dive_trigger_mode', 'distance_xyz')
    ms['cruise'].setdefault('dive_altitude_threshold_m', 1000.0)

    ms.setdefault('aircraft', {})
    ms['aircraft'].setdefault('turn_rate_rad_s', 0.02)
    ms['aircraft'].setdefault('speed_jitter_std', 0.0)

    cfg.setdefault('tti_solver', {})
    ts = cfg['tti_solver']
    # TTI 求解器步长与最大时间，较小配置可显著降低计算量
    ts.setdefault('dt', 2.0)
    ts.setdefault('max_time', 120.0)
    ts.setdefault('mode', 'deterministic')
    ts.setdefault('mc_samples', 20)
    ts.setdefault('mc_noise_std', 50.0)
    ts.setdefault('aim_mode', 'predicted_future')
    return cfg


def choose_actions(env: WTAEnv):
    """根据掩码选择合法分配：每拦截器最多分配1个，优先满足防御窗与射程。"""
    masks = env.get_action_masks()
    alive = masks.get('alive_mask')
    rng_mask = masks.get('range_mask')
    cap_mask = masks.get('capacity_mask')
    def_mask = masks.get('defense_time_mask')

    if alive is None or rng_mask is None or cap_mask is None or def_mask is None:
        return []

    num_i = env.state.get('num_interceptors', 0)
    num_j = env.state.get('num_targets', 0)
    pairs = []
    used_targets = set()
    used_interceptors = set()

    # 合并掩码：必须同时满足
    combined = alive & rng_mask & cap_mask & def_mask
    # 贪心：按 TTI 从小到大尝试分配（复用掩码中已计算的 TTI，避免重复重算）
    tti = masks.get('pairwise_tti')
    if tti is None:
        tti = env.engine.pairwise_tti(env.state['interceptors'], env.state['targets'])
    idxs = np.argwhere(combined)
    idxs_sorted = sorted(idxs.tolist(), key=lambda ij: tti[ij[0], ij[1]])
    for i, j in idxs_sorted:
        if i in used_interceptors:
            continue
        if j in used_targets:
            continue
        pairs.append((int(i), int(j)))
        used_interceptors.add(int(i))
        used_targets.add(int(j))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--env', type=str, default='config/env.yaml')
    args = parser.parse_args()

    env_cfg = load_env_cfg(args.env)
    reward_cfg = {}
    model_cfg = {}

    env = WTAEnv(env_cfg, reward_cfg, model_cfg)
    obs0 = env.reset()
    print(f"Reset done. t={env.t}, targets={env.state['num_targets']}, interceptors={env.state['num_interceptors']}")
    print(f"Pairwise distance shape: {obs0['pairwise_distance'].shape}, Pairwise TTI shape: {obs0['pairwise_tti'].shape}")

    for step in range(args.steps):
        pairs = choose_actions(env)
        obs, reward, done, info = env.step(pairs)
        print(f"Step {step+1}/{args.steps}: assignments={pairs}, shots={info['shots']}, hits={info['hits']}, misses={info['misses']}, pending={info['pending_events']}")
        if done:
            print("Episode finished early.")
            break

    print("Demo finished.")


if __name__ == '__main__':
    main()