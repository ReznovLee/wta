#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
WTAEnv: 多目标-多拦截器的最小可运行环境（支持一步多分配），按用户偏好增强：

增强点概览：
- 动作约束：每步每拦截器最多分配1个目标；每步每目标的可被分配容量由目标类型决定：
  - ballistic（弹道导弹）: 3；
  - cruise（巡航导弹）: 1；
  - aircraft（飞机）: 1。
- TTI/防御窗：若估计的拦截时间 TTI 超过目标到达防御点（或防御区）的时间，则拒绝该分配并记录为 late_assignments，不入队事件。
- RNG：接入 rng 保证可复现（默认 seed=env_cfg.seed 或 42）。
- 掩码：在基础掩码外，加入目标存活掩码、容量掩码、射程掩码；
- 观测：补充 pairwise 距离矩阵与 TTI 估计矩阵，加入目标类型id；
- 历史记录：保存每步的分配与结果，便于指标统计与数据集生成。

依赖：
- PhysicsEngine：提供 in_range(...)、estimate_intercept_time(...)、propagate(...)、pairwise_distance(...)、pairwise_tti(...)、time_to_defended_zone(...)
- RewardCalculator：提供 compute_step_reward(assignments, outcomes, dt) 与 compute_terminal_reward(summary)
"""
import numpy as np
import os
import csv
from .physics_engine import PhysicsEngine
from .reward_calculator import RewardCalculator
from .action_mask import build_action_masks


class WTAEnv:
    """支持一步多分配与TTI事件队列的 WTA 环境（按目标类型容量约束）。"""

    def __init__(self, env_cfg, reward_cfg, model_cfg):
        # 配置与引擎
        self.env_cfg = env_cfg or {}
        self.reward_calc = RewardCalculator(reward_cfg or {})
        self.engine = PhysicsEngine(env_cfg or {})

        # RNG（可复现）
        self.seed = int(self.env_cfg.get('seed', 42))
        self.rng = np.random.RandomState(self.seed)

        # 时间控制
        self.time_step = float(self.env_cfg.get('time_step', 1.0))
        self.horizon = int(self.env_cfg.get('horizon', 200))
        self.t = 0.0  # 连续时间（也允许非整数）

        # 目标类型容量约束（可在 env.yaml.capacity_by_type 覆盖）
        self.capacity_by_type = {
            'ballistic': int(self.env_cfg.get('capacity_by_type', {}).get('ballistic', 3)),
            'cruise': int(self.env_cfg.get('capacity_by_type', {}).get('cruise', 1)),
            'aircraft': int(self.env_cfg.get('capacity_by_type', {}).get('aircraft', 1)),
        }

        # 状态与事件
        self.state = None
        self._events = []  # 事件队列：[{ 'i': int, 'j': int, 'scheduled_time': float, 'hit_prob': float }]
        self.history = []  # 每步历史记录

    # ------------------------------
    # 环境主流程
    # ------------------------------
    def reset(self):
        """初始化目标与拦截器集合，清空事件队列并返回初始观测（使用 rng 保证可复现）。"""
        print("DEBUG: WTAEnv.reset called", flush=True)
        num_targets = int(self.env_cfg.get('max_targets', 20))
        num_interceptors = int(self.env_cfg.get('max_interceptors', 10))
        # 优先：调用场景生成器生成更完整、可复现的初始化（失败则回退到 CSV/随机）
        targets = []
        interceptors = []
        try:
            print("DEBUG: Importing make_scenario", flush=True)
            from .scenario_generator import make_scenario
            print("DEBUG: Calling make_scenario", flush=True)
            scenario = make_scenario(self.env_cfg, num_targets=num_targets, num_interceptors=num_interceptors, seed=self.seed)
            print("DEBUG: make_scenario returned", flush=True)
            if isinstance(scenario, dict):
                sg_targets = scenario.get('targets', [])
                sg_interceptors = scenario.get('interceptors', [])
                if sg_targets and sg_interceptors:
                    targets = sg_targets
                    interceptors = sg_interceptors
                    # 用场景生成器补齐后的 env_cfg 覆盖当前配置，并重建引擎以应用最新配置
                    new_env_cfg = scenario.get('env_cfg', self.env_cfg)
                    if isinstance(new_env_cfg, dict):
                        self.env_cfg = new_env_cfg
                        # 若种子发生变化，更新 RNG
                        new_seed = int(self.env_cfg.get('seed', self.seed))
                        if new_seed != self.seed:
                            self.set_seed(new_seed)
                        # 重建 PhysicsEngine 以应用 tti_solver/motion_specs 等最新配置
                        print("DEBUG: Re-initializing PhysicsEngine", flush=True)
                        self.engine = PhysicsEngine(self.env_cfg)
                        print("DEBUG: PhysicsEngine re-initialized", flush=True)
        except Exception as e:
            print(f"DEBUG: Exception in reset scenario generation: {e}", flush=True)
            # 场景生成器不可用时，继续采用 CSV/随机回退
            pass

        # 目标类型分布（可在 env.yaml.target_type_distribution 配置），默认等概率
        type_dist = self.env_cfg.get('target_type_distribution', {
            'ballistic': 1/3, 'cruise': 1/3, 'aircraft': 1/3
        })
        type_names = ['ballistic', 'cruise', 'aircraft']
        probs = np.array([float(type_dist.get(k, 0)) for k in type_names], dtype=float)
        if probs.sum() <= 0:
            probs = np.array([1/3, 1/3, 1/3], dtype=float)
        probs = probs / probs.sum()

        # 若场景生成器未提供目标，则尝试 CSV；仍为空时使用占位随机初始化
        if not targets:
            targets = self._load_targets_csv(self.env_cfg.get('targets_csv'))
        if not targets:
            for _ in range(num_targets):
                ttype = self.rng.choice(type_names, p=probs)
                tgt = {
                    'position': self.rng.uniform(-10000, 10000, size=3),
                    'velocity': self.rng.uniform(-50, 50, size=3),
                    'value': float(self.rng.uniform(0.5, 1.0)),
                    'threat': float(self.rng.uniform(0.5, 1.0)),
                    'type': ttype,
                    'alive': True,
                }
                targets.append(tgt)
        num_targets = len(targets)

        # 武器规格默认值（env.yaml.weapon_specs 可覆盖）
        weapon_specs = self.env_cfg.get('weapon_specs', {})
        default_ammo = int(weapon_specs.get('ammo_per_unit', 4))
        default_speed = float(weapon_specs.get('speed_mps', 300.0))
        default_hit_prob = float(weapon_specs.get('base_hit_prob', 0.7))

        # 若场景生成器未提供拦截器，则尝试 CSV；仍为空时使用占位随机初始化
        if not interceptors:
            interceptors = self._load_interceptors_csv(self.env_cfg.get('interceptors_csv'))
        if not interceptors:
            for _ in range(num_interceptors):
                itc = {
                    'position': self.rng.uniform(-10000, 10000, size=3),
                    'speed': default_speed,
                    'ammo': default_ammo,
                    'hit_prob': default_hit_prob,
                }
                interceptors.append(itc)
        num_interceptors = len(interceptors)

        self.state = {
            'targets': targets,
            'interceptors': interceptors,
            'num_targets': num_targets,
            'num_interceptors': num_interceptors,
            'interceptor_ammo': [it['ammo'] for it in interceptors],
        }
        self._events = []
        self.history = []
        self.t = 0.0
        print("DEBUG: reset calling compute_observation", flush=True)
        obs = self.compute_observation(self.state, self.t)
        print("DEBUG: reset compute_observation returned", flush=True)
        return obs

    def step(self, action):
        """
        执行一步仿真（动力学版）：
        - 解析动作，激活拦截器进入飞行状态（status='flying'）；
        - 推进物理引擎（目标+拦截器）；
        - 检查拦截器状态（hit/miss）并结算。
        """
        outcomes = {
            'hits': 0,
            'misses': 0,
            'violations': 0,
            'shots': 0,
            'delay': 0.0,
            'late_assignments': 0,
            'capacity_violations': 0,
            'extra_assignments': 0,
            'hits_value_sum': 0.0,
            'time_factor_sum': 0.0,
            'coop_met_hits': 0,
        }

        # 1) 解析动作为分配对列表 pairs
        pairs = self._normalize_action(action)

        # 每步每拦截器最多分配1个目标
        filtered_pairs = []
        used_interceptors = set()
        for (i, j) in pairs:
            if i in used_interceptors:
                outcomes['extra_assignments'] += 1
                continue
            used_interceptors.add(i)
            filtered_pairs.append((i, j))
        pairs = filtered_pairs

        # 2) 处理新发射
        for (i, j) in pairs:
            if not self._is_valid_index(i, j):
                outcomes['violations'] += 1
                continue
            
            interceptor = self.state['interceptors'][i]
            target = self.state['targets'][j]
            
            # 只有闲置且有弹药的拦截器才能发射
            if interceptor.get('status', 'idle') != 'idle':
                # 已经在飞的不能重新分配（除非实现重制导，这里假设发射后不管）
                outcomes['violations'] += 1
                continue

            if not target.get('alive', True):
                outcomes['violations'] += 1
                continue

            # 容量约束 check
            target_id = target.get('id')
            ttype = str(target.get('type', 'cruise'))
            cap = int(self.capacity_by_type.get(ttype, 1))
            # 统计当前已经在飞的 + 本次这一步刚刚分配的（如果支持单步对同一目标多发）
            # 注意：pairs 是处理后的，不会有重复 i，但可能有多个 i 指向同一个 j
            flying_count = sum(1 for itc in self.state['interceptors'] 
                             if itc.get('status') == 'flying' and itc.get('target_id') == target_id)
            # 加上本次 step 中前面已经处理的指向该 target 的
            # (这里简单处理，假设 batch 动作符合掩码约束，如果违反则记录)
            # 为严格起见，我们可以再统计一下 pairs 里的
            current_step_count = sum(1 for (pi, pj) in pairs if pj == j and pi != i) # 粗略估计
            
            if flying_count >= cap:
                outcomes['capacity_violations'] += 1
                continue

            # 射程与弹药校验
            in_range = self.engine.in_range(interceptor, target)
            has_ammo = interceptor['ammo'] > 0
            if not (in_range and has_ammo):
                outcomes['violations'] += 1
                continue

            # TTI 预估校验（可选，作为智能体的辅助限制）
            tti_est = float(self.engine.estimate_intercept_time(interceptor, target))
            t_defense = float(self.engine.time_to_defended_zone(target))
            if np.isfinite(t_defense) and tti_est > t_defense:
                outcomes['late_assignments'] += 1
                continue

            # --- 激活拦截器 ---
            interceptor['status'] = 'flying'
            interceptor['target_id'] = target_id
            interceptor['ammo'] -= 1
            
            # 初始化速度矢量：指向目标当前位置
            p_i = np.array(interceptor.get('position', [0,0,0]), dtype=float)
            p_t = np.array(target.get('position', [0,0,0]), dtype=float)
            speed = float(interceptor.get('speed', 300.0))
            dir_vec = p_t - p_i
            dist = np.linalg.norm(dir_vec)
            if dist > 1e-9:
                vel = dir_vec / dist * speed
            else:
                vel = np.array([0, 0, speed], dtype=float) # 默认向上?
            interceptor['velocity'] = vel
            
            # 记录发射
            outcomes['shots'] += 1
            # 记录 launch time 方便计算 delay
            interceptor['launch_time'] = self.t
            interceptor['launch_tti_est'] = tti_est # 记录预估，用于分析

        # 3) 物理推进
        # Pass interceptors explicitly!
        self.engine.propagate(self.state['targets'], self.state['interceptors'], dt=self.time_step)
        self.t += self.time_step

        # 4) 结算结果
        for itc in self.state['interceptors']:
            if itc.get('status') == 'hit':
                # 命中
                tid = itc.get('target_id')
                # 找到目标
                tgt = next((t for t in self.state['targets'] if t.get('id') == tid), None)
                if tgt and tgt.get('alive', True):
                    # 概率判定 (hit_prob)
                    # 动力学已经判定了“接近”，这里判定“引信/毁伤”概率
                    prob = float(itc.get('hit_prob', 0.7))
                    if self.rng.rand() < prob:
                        outcomes['hits'] += 1
                        tgt['alive'] = False
                        
                        # 价值统计
                        vmap = {'ballistic': 3.0, 'cruise': 2.0, 'aircraft': 1.0}
                        vt = float(vmap.get(str(tgt.get('type', 'cruise')), 1.0))
                        outcomes['hits_value_sum'] += vt
                        
                        # 时间因子
                        t_defense = float(self.engine.time_to_defended_zone(tgt))
                        if np.isfinite(t_defense) and t_defense > 1e-9:
                             # 粗略计算：越早拦截越好
                             # 注意：t_defense 是基于当前位置的剩余时间。
                             # 若要精确计算需记录初始 t_defense。这里简化处理。
                             tf = 1.0 # 简化
                             outcomes['time_factor_sum'] += tf
                        
                        # 协同统计
                        # 统计有多少枚拦截器也在攻击这个目标（不论是否命中，只要是 flying 或刚刚 hit）
                        # 此时该拦截器状态是 hit。
                        # 我们可以统计“同时命中”或“饱和攻击”。
                        # 简单起见：
                        coop_min = int(self.env_cfg.get('coop_min_by_type', {}).get(str(tgt.get('type', 'cruise')), 1))
                        # 实际上很难精确统计“协同”，因为是异步到达。
                        # 这里暂且计为 1
                        if coop_min <= 1:
                            outcomes['coop_met_hits'] += 1
                    else:
                        outcomes['misses'] += 1
                
                # Reset interceptor
                itc['status'] = 'idle'
                itc['target_id'] = None
                itc['velocity'] = np.zeros(3)
                
            elif itc.get('status') == 'miss':
                outcomes['misses'] += 1
                itc['status'] = 'idle'
                itc['target_id'] = None
                itc['velocity'] = np.zeros(3)

        # 5) 计算观测与终止条件
        obs = self.compute_observation(self.state, self.t)
        reward = self.reward_calc.compute_step_reward(None, outcomes, dt=self.time_step)

        # 是否所有目标已被拦截
        all_intercepted = all((not t.get('alive', True)) for t in self.state['targets'])
        done = (self.t >= self.horizon) or all_intercepted
        if done:
            alive_vals = [float({'ballistic': 3.0, 'cruise': 2.0, 'aircraft': 1.0}.get(str(t.get('type', 'cruise')), 1.0)) for t in self.state['targets'] if t.get('alive', True)]
            reward += self.reward_calc.compute_terminal_reward({'intercept_all': all_intercepted, 'penetrated_value_sum': float(sum(alive_vals))})

        info = {
            't': self.t,
            'shots': outcomes['shots'],
            'hits': outcomes['hits'],
            'misses': outcomes['misses'],
            'violations': outcomes['violations'],
            'late_assignments': outcomes['late_assignments'],
            'capacity_violations': outcomes['capacity_violations'],
            'extra_assignments': outcomes['extra_assignments'],
            'delay': outcomes['delay'],
            'assignments': pairs,
        }
        # 记录历史
        self.history.append({
            't': self.t,
            'assignments': pairs,
            'outcomes': outcomes,
        })
        return obs, reward, done, info

    # ------------------------------
    # 掩码与观测
    # ------------------------------
    def get_action_masks(self, state=None):
        """
        返回动作约束掩码：基础掩码来自 build_action_masks，并额外提供目标存活掩码。
        """
        s = state or self.state
        base_masks = build_action_masks(s, self.env_cfg)
        # 目标存活约束：不可对已拦截的目标进行分配
        num_i = s.get('num_interceptors', 0)
        num_j = s.get('num_targets', 0)
        alive = np.array([int(t.get('alive', True)) for t in s.get('targets', [])], dtype=bool)
        alive_mask = np.tile(alive, (num_i, 1)) if num_j > 0 else np.zeros((num_i, 0), dtype=bool)
        base_masks['alive_mask'] = alive_mask

        # 射程掩码：当前时刻拦截器-目标对是否在射程内
        range_mask = np.zeros((num_i, num_j), dtype=bool)
        for i in range(num_i):
            for j in range(num_j):
                range_mask[i, j] = self.engine.in_range(s['interceptors'][i], s['targets'][j])
        base_masks['range_mask'] = range_mask

        # 容量掩码：考虑已在队列中的 pending 事件，若某目标已达到容量则该列全 False
        capacity_mask = np.ones((num_i, num_j), dtype=bool)
        for j in range(num_j):
            ttype = str(s['targets'][j].get('type', 'cruise'))
            cap = int(self.capacity_by_type.get(ttype, 1))
            pending_to_target = sum(1 for ev in self._events if ev.get('j') == j)
            if pending_to_target >= cap:
                capacity_mask[:, j] = False
        base_masks['capacity_mask'] = capacity_mask

        # 防御窗掩码：TTI <= 目标抵达防御区时间
        # 注意：pairwise_tti 计算较为耗时，此处已计算一次，供外部复用，避免重复计算
        pair_tti = self.engine.pairwise_tti(s['interceptors'], s['targets'])
        t_defense = np.array([self.engine.time_to_defended_zone(t) for t in s['targets']], dtype=float)
        defense_time_mask = np.zeros((num_i, num_j), dtype=bool)
        slack = float(self.env_cfg.get('defense_time_slack', 0.2))
        for i in range(num_i):
            for j in range(num_j):
                td = t_defense[j]
                tij = pair_tti[i, j]
                defense_time_mask[i, j] = np.isfinite(td) and np.isfinite(tij) and (tij <= td * (1.0 + slack))
        base_masks['defense_time_mask'] = defense_time_mask
        # 将已经计算好的 pairwise_tti 暴露出去，便于上层策略/脚本直接复用
        base_masks['pairwise_tti'] = pair_tti
        return base_masks

    def compute_observation(self, state, t):
        """
        观测结构（字典）：
        - 目标：位置、速度、价值、威胁度、存活标记
        - 拦截器：位置、弹药、命中概率、速度
        - 时间：当前仿真时刻 t
        注意：保持维度稳定，存活标记用于下游模型或掩码结合。
        """
        print("DEBUG: compute_observation start", flush=True)
        tgt = state['targets']
        itc = state['interceptors']
        # 目标类型编码：ballistic=0, cruise=1, aircraft=2
        type_to_id = {'ballistic': 0, 'cruise': 1, 'aircraft': 2}

        # 计算 pairwise 距离与 TTI 估计（用于模型与掩码）
        print("DEBUG: calling pairwise_distance", flush=True)
        pair_dist = self.engine.pairwise_distance(state['interceptors'], state['targets'])
        print("DEBUG: calling pairwise_tti", flush=True)
        pair_tti = self.engine.pairwise_tti(state['interceptors'], state['targets'])
        print("DEBUG: pairwise calcs done", flush=True)

        obs = {
            'targets_pos': np.array([x['position'] for x in tgt]) if tgt else np.zeros((0, 3)),
            'targets_vel': np.array([x['velocity'] for x in tgt]) if tgt else np.zeros((0, 3)),
            'targets_val': np.array([x['value'] for x in tgt]) if tgt else np.zeros((0,)),
            'targets_thr': np.array([x['threat'] for x in tgt]) if tgt else np.zeros((0,)),
            'targets_alive': np.array([x.get('alive', True) for x in tgt], dtype=bool) if tgt else np.zeros((0,), dtype=bool),
            'targets_type_id': np.array([type_to_id.get(str(x.get('type', 'cruise')), 1) for x in tgt], dtype=int) if tgt else np.zeros((0,), dtype=int),
            'interceptors_pos': np.array([x['position'] for x in itc]) if itc else np.zeros((0, 3)),
            'interceptors_ammo': np.array([x['ammo'] for x in itc]) if itc else np.zeros((0,)),
            'interceptors_hit_prob': np.array([x.get('hit_prob', 0.7) for x in itc]) if itc else np.zeros((0,)),
            'interceptors_speed': np.array([x.get('speed', 300.0) for x in itc]) if itc else np.zeros((0,)),
            't': float(t),
            'pairwise_distance': pair_dist,
            'pairwise_tti': pair_tti,
        }
        return obs

    # ------------------------------
    # 工具方法
    # ------------------------------
    def _normalize_action(self, action):
        """将动作统一为分配对列表 [(i, j), ...]。"""
        pairs = []
        if action is None:
            return pairs
        # 矩阵格式
        if isinstance(action, (np.ndarray, list)) and not (len(action) > 0 and isinstance(action[0], tuple)):
            mat = np.array(action)
            if mat.ndim != 2:
                return pairs
            num_i = self.state.get('num_interceptors', 0)
            num_j = self.state.get('num_targets', 0)
            mi, mj = mat.shape
            if mi != num_i or mj != num_j:
                # 维度不匹配视为违例，不抛异常，保持稳健
                return pairs
            idxs = np.argwhere(mat.astype(bool))
            for i, j in idxs:
                pairs.append((int(i), int(j)))
            return pairs
        # 列表对格式
        if isinstance(action, list):
            for item in action:
                if isinstance(item, tuple) and len(item) == 2:
                    i, j = int(item[0]), int(item[1])
                    pairs.append((i, j))
        return pairs

    def _is_valid_index(self, i, j):
        return (0 <= i < self.state.get('num_interceptors', 0)) and (0 <= j < self.state.get('num_targets', 0))

    # ------------------------------
    # 种子控制
    # ------------------------------
    def set_seed(self, seed: int):
        """设置随机种子，重置 rng。"""
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

    # ------------------------------
    # CSV 加载辅助
    # ------------------------------
    def _load_interceptors_csv(self, path):
        items = []
        try:
            if path and os.path.exists(path):
                with open(path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        pos = [float(row.get('x', 0)), float(row.get('y', 0)), float(row.get('z', 0))]
                        item = {
                            'position': np.array(pos, dtype=float),
                            'speed': float(row.get('speed', self.env_cfg.get('weapon_specs', {}).get('speed_mps', 300.0))),
                            'ammo': int(row.get('ammo', self.env_cfg.get('weapon_specs', {}).get('ammo_per_unit', 4))),
                            'hit_prob': float(row.get('hit_prob', self.env_cfg.get('weapon_specs', {}).get('base_hit_prob', 0.7))),
                        }
                        items.append(item)
        except Exception:
            return []
        return items

    def _load_targets_csv(self, path):
        items = []
        try:
            if path and os.path.exists(path):
                with open(path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        pos = [float(row.get('x', 0)), float(row.get('y', 0)), float(row.get('z', 0))]
                        vel = [float(row.get('vx', 0)), float(row.get('vy', 0)), float(row.get('vz', 0))]
                        ttype = str(row.get('type', 'cruise'))
                        item = {
                            'position': np.array(pos, dtype=float),
                            'velocity': np.array(vel, dtype=float),
                            'value': float(row.get('value', 1.0)),
                            'threat': float(row.get('threat', 1.0)),
                            'type': ttype,
                            'alive': True,
                        }
                        items.append(item)
        except Exception:
            return []
        return items
