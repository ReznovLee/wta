import csv
import os
from typing import List, Dict, Optional

import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class PhysicsEngine:
    """
    物理引擎（更新版）：
    - 目标运动模型按你的描述实现：
      * ballistic（弹道）：进入防守空域后为斜抛运动的后半段（下落段），围绕已知落点（impact_point）存在散布；可叠加轻微速度抖动。
      * cruise（巡航）：先做与地面平行的曲线运动（横向扰动），接近落点（impact_point）时切换为俯冲阶段；可叠加速度抖动。
      * aircraft（飞机）：单纯的曲线运动（恒定角速度水平转向，速度模长恒定，可选小抖动）。
    - 拦截弹：直线近似（固定朝向“当前目标位置或其已知落点”的方向单位向量 u_hat），以速度 v_i 匀速运动。
    - TTI：采用数值搜索，目标未来位置由“扩展卡尔曼滤波（EKF）接口”或内置模型模拟得到；拦截器直线推进，首次满足距离小于拦截半径即为 TTI；超出防御窗则视为不可拦截。
    - 目标到达防御区时间：数值步进，按上述模型/接口推进目标位置，首次进入防御区半径返回时间。
    - 成对距离与射程判断维持不变；支持从 CSV 读取防御点。

    可配置项（env_cfg）：
      weapon_specs: { max_range_km, speed_mps, intercept_radius_m }
      defended_zone: { center: [x,y,z], radius_m, center_mode: 'fixed' | 'nearest' }
      motion_specs:
        ballistic: { gravity_mps2, scatter_sigma_m, speed_jitter_std }
        cruise: { lateral_amp_m, lateral_omega, dive_threshold_m, dive_rate_mps, speed_jitter_std,
                  dive_trigger_mode: 'distance_xyz' | 'distance_xy' | 'altitude',
                  dive_altitude_threshold_m }
        aircraft: { turn_rate_rad_s, speed_jitter_std }
      tti_solver: { dt, max_time, mode: 'deterministic' | 'monte_carlo', mc_samples, mc_noise_std,
                    aim_mode: 'impact_point' | 'defense_center' | 'predicted_future' }
      seed: 用于引擎内部随机抖动的可复现性

    EKF 接口约定（由 src/utils/filter.py 提供实现）：
      - EKFTracker 类：
          predict_horizon(current_pos: np.ndarray, current_vel: np.ndarray, tau: float, target_meta: Dict) -> Dict
        返回 { 'position': np.ndarray(shape=(3,)), 'velocity': np.ndarray(shape=(3,)) }
      - 目标字典可包含 'tracker'（EKFTracker 实例）或 'tracker_state'（初始化状态），引擎会调用其接口进行未来预测。
    """

    def __init__(self, env_cfg: Dict):
        self.env_cfg = env_cfg or {}
        self.weapon_specs = self.env_cfg.get('weapon_specs', {})
        self.defended_zone = self.env_cfg.get('defended_zone', {})
        self.motion_specs = self.env_cfg.get('motion_specs', {})
        self.tti_solver = self.env_cfg.get('tti_solver', {})
        self.fast_mode = bool(self.tti_solver.get('fast_mode', False))
        try:
            self.use_cuda = bool(self.env_cfg.get('use_cuda', False)) and TORCH_AVAILABLE and torch.cuda.is_available()
        except Exception:
            self.use_cuda = False

        # RNG（用于速度抖动与散布，可复现）
        self.seed = int(self.env_cfg.get('seed', 42))
        self.rng = np.random.RandomState(self.seed)

        # Solver params
        self.dt_solver = float(self.tti_solver.get('dt', 0.5))        # s
        self.max_time_solver = float(self.tti_solver.get('max_time', 600.0))  # s
        # TTI 模式配置（deterministic 或 monte_carlo）
        self.tti_mode = str(self.tti_solver.get('mode', 'deterministic'))
        self.mc_samples = int(self.tti_solver.get('mc_samples', 20))
        self.mc_noise_std = float(self.tti_solver.get('mc_noise_std', 50.0))  # m
        # 拦截器瞄准模式：impact_point | defense_center | predicted_future
        self.aim_mode = str(self.tti_solver.get('aim_mode', 'predicted_future'))

        # Guidance constants
        self.pn_constant = float(self.weapon_specs.get('pn_constant', 3.0))
        self.max_acc_g = float(self.weapon_specs.get('max_acc_g', 20.0))


        # 防御点 CSV（可选）
        self.defense_points = self._load_defense_points_csv(
            self.env_cfg.get('defense_points_csv', os.path.join('data', 'defense_points.csv'))
        )

        # 轨迹 JSON（可选）：每目标时序采样数据，用于严格复现
        self.trajectories = self._load_trajectories_json(self.env_cfg.get('trajectories_json'))

    # ------------------------------
    # IO: 读取防御点 CSV
    # ------------------------------
    def _load_defense_points_csv(self, path: str) -> List[Dict]:
        pts = []
        try:
            if path and os.path.exists(path):
                with open(path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'x' in row and 'y' in row and 'z' in row:
                            name = row.get('name', f"pt_{len(pts)}")
                            pts.append({
                                'name': name,
                                'position': np.array([
                                    float(row['x']), float(row['y']), float(row['z'])
                                ], dtype=float)
                            })
        except Exception:
            return []
        return pts

    def _load_trajectories_json(self, path: Optional[str]) -> Dict:
        """可选：从 JSON 加载目标时序轨迹，格式建议：
        {
          "targets": [
             {"id": 0, "samples": [{"t": 0.0, "pos": [..], "vel": [..]}, ...], "type": "cruise"}, ...
          ]
        }
        返回以 id 为键的字典；若无文件或解析失败则返回空字典。
        """
        if not path or not os.path.exists(path):
            return {}
        try:
            import json
            with open(path, 'r') as f:
                data = json.load(f)
            result = {}
            for item in data.get('targets', []):
                tid = int(item.get('id', len(result)))
                result[tid] = item
            return result
        except Exception:
            return {}

    # ------------------------------
    # 运动推进（一步）
    # ------------------------------
    def propagate(self, targets: List[Dict], interceptors: List[Dict], dt: float):
        # 1. Update Targets
        for t in targets:
            # 若存在全局 trajectories 映射且目标未绑定轨迹，按 id 自动绑定一次
            tid = t.get('id', None)
            if tid is not None and 'trajectory' not in t and isinstance(self.trajectories, dict) and tid in self.trajectories:
                t['trajectory'] = self.trajectories[tid]
                t['sample_index'] = int(t.get('sample_index', 0))
                t['current_time'] = float(t.get('current_time', 0.0))
            # 若从 JSON 载入了时序轨迹，优先使用轨迹采样推进
            if 'trajectory' in t and isinstance(t['trajectory'], dict) and 'samples' in t['trajectory']:
                samples = t['trajectory']['samples']
                # 若采样包含时间戳，则进行时间驱动的推进与线性插值；否则退化为索引推进
                has_time = len(samples) > 0 and isinstance(samples[0], dict) and ('t' in samples[0])
                if has_time:
                    tcur = float(t.get('current_time', 0.0))
                    tnext = tcur + float(dt)
                    idx = int(t.get('sample_index', 0))
                    # 找到第一个 samples[k]['t'] >= tnext
                    k = idx
                    while k < len(samples) and float(samples[k].get('t', 0.0)) < tnext:
                        k += 1
                    if k >= len(samples):
                        # 超过最后一个采样，钳制为最后一帧
                        k = len(samples) - 1
                        s_hi = samples[k]
                        t['position'] = np.array(s_hi.get('pos', t.get('position', [0, 0, 0])), dtype=float)
                        t['velocity'] = np.array(s_hi.get('vel', t.get('velocity', [0, 0, 0])), dtype=float)
                        t['type'] = s_hi.get('type', t.get('type', 'cruise'))
                    else:
                        s_hi = samples[k]
                        if k == 0:
                            s_lo = s_hi
                        else:
                            s_lo = samples[k - 1]
                        t_lo = float(s_lo.get('t', 0.0))
                        t_hi = float(s_hi.get('t', t_lo))
                        # 线性插值
                        if t_hi > t_lo:
                            alpha = max(0.0, min(1.0, (tnext - t_lo) / (t_hi - t_lo)))
                        else:
                            alpha = 1.0
                        pos_lo = np.array(s_lo.get('pos', t.get('position', [0, 0, 0])), dtype=float)
                        pos_hi = np.array(s_hi.get('pos', t.get('position', [0, 0, 0])), dtype=float)
                        vel_lo = np.array(s_lo.get('vel', t.get('velocity', [0, 0, 0])), dtype=float)
                        vel_hi = np.array(s_hi.get('vel', t.get('velocity', [0, 0, 0])), dtype=float)
                        pos_interp = (1.0 - alpha) * pos_lo + alpha * pos_hi
                        vel_interp = (1.0 - alpha) * vel_lo + alpha * vel_hi
                        t['position'] = pos_interp
                        t['velocity'] = vel_interp
                        t['type'] = s_hi.get('type', t.get('type', 'cruise'))
                        # 更新索引
                        t['sample_index'] = k
                    t['current_time'] = tnext
                else:
                    # 无时间戳，逐索引推进
                    idx = int(t.get('sample_index', 0))
                    next_idx = min(idx + 1, len(samples) - 1)
                    t['sample_index'] = next_idx
                    s = samples[next_idx]
                    t['position'] = np.array(s.get('pos', t.get('position', [0, 0, 0])), dtype=float)
                    t['velocity'] = np.array(s.get('vel', t.get('velocity', [0, 0, 0])), dtype=float)
                    t['type'] = s.get('type', t.get('type', 'cruise'))
            else:
                # 模型推进
                self._advance_target_state(t, dt)

        # 2. Update Interceptors
        for itc in interceptors:
            if itc.get('status') == 'flying':
                target_id = itc.get('target_id')
                # Find target
                target = None
                for t in targets:
                    if t.get('id') == target_id:
                        target = t
                        break
                
                if target and target.get('alive', True):
                    self._update_interceptor_dynamics(itc, target, dt)
                else:
                    # Target lost or dead
                    itc['status'] = 'miss'

    def _step_interceptor_dynamics_pure(self, pos_m: np.ndarray, vel_m: np.ndarray, 
                                      pos_t: np.ndarray, vel_t: np.ndarray, 
                                      dt: float, speed_limit: float) -> (np.ndarray, np.ndarray):
        """
        纯函数形式的拦截器动力学步进 (PN 制导)
        返回: (next_position, next_velocity)
        """
        # Constants
        N_pn = self.pn_constant
        max_acc_g = self.max_acc_g
        g = np.array([0, 0, -9.81])
        
        # Relative states
        r_vec = pos_t - pos_m
        r = np.linalg.norm(r_vec)
        # Avoid division by zero
        if r < 1e-3:
            return pos_m, vel_m

        v_rel = vel_t - vel_m 
        
        # PN Guidance Law
        cross_prod = np.cross(r_vec, v_rel)
        r2 = r**2 + 1e-9
        omega_vec = cross_prod / r2
        
        acc_cmd = N_pn * np.cross(v_rel, omega_vec)
        
        # Limit acceleration
        max_acc = max_acc_g * 9.81
        acc_norm = np.linalg.norm(acc_cmd)
        if acc_norm > max_acc:
            acc_cmd = acc_cmd / acc_norm * max_acc
            
        # Update Velocity (Euler integration)
        acc_total = acc_cmd + g
        vel_new = vel_m + acc_total * dt
        
        # Enforce constant speed
        v_norm = np.linalg.norm(vel_new)
        if v_norm > 1e-9:
            vel_new = vel_new / v_norm * speed_limit
            
        pos_new = pos_m + vel_new * dt
        return pos_new, vel_new

    def _update_interceptor_dynamics(self, interceptor: Dict, target: Dict, dt: float):
        # Constants from config
        pos_m = np.array(interceptor.get('position', [0,0,0]), dtype=float)
        vel_m = np.array(interceptor.get('velocity', [0,0,0]), dtype=float)
        pos_t = np.array(target.get('position', [0,0,0]), dtype=float)
        vel_t = np.array(target.get('velocity', [0,0,0]), dtype=float)
        speed = float(interceptor.get('speed', 300.0))
        max_flight_time = float(self.weapon_specs.get('max_flight_time_s', 120.0))

        # Track flight time
        current_flight_time = float(interceptor.get('flight_time', 0.0))
        current_flight_time += dt
        interceptor['flight_time'] = current_flight_time

        if current_flight_time > max_flight_time:
            interceptor['status'] = 'miss'
            return
            
        # Track flight distance
        vel_norm = np.linalg.norm(vel_m)
        dist_step = vel_norm * dt
        current_flight_dist = float(interceptor.get('flight_dist', 0.0))
        current_flight_dist += dist_step
        interceptor['flight_dist'] = current_flight_dist
        
        max_range_m = float(self.weapon_specs.get('max_range_km', 100.0)) * 1000.0
        if current_flight_dist > max_range_m:
             interceptor['status'] = 'miss'
             return

        
        # Check hit logic is handled outside or before pure dynamics
        # Here we just check radius for 'hit' status update if not already handled
        r_vec = pos_t - pos_m
        r = np.linalg.norm(r_vec)
        radius = float(self.weapon_specs.get('intercept_radius_m', 50.0))
        
        if r <= radius:
            interceptor['status'] = 'hit'
            interceptor['position'] = pos_t 
            return

        # Use pure dynamics
        pos_new, vel_new = self._step_interceptor_dynamics_pure(pos_m, vel_m, pos_t, vel_t, dt, speed)
        
        # Update state
        interceptor['position'] = pos_new
        interceptor['velocity'] = vel_new
        
        # Check miss condition: moving away from target
        # Re-calculate r_vec, v_rel with new states or old? 
        # Usually check based on current state before update, or after.
        # Let's keep consistency with old logic: check miss based on state BEFORE update?
        # Actually, the old logic used r_vec (old) and v_rel (old).
        v_rel = vel_t - vel_m
        if np.dot(r_vec, v_rel) > 0 and r > radius * 3.0:
             interceptor['status'] = 'miss'

    # ------------------------------
    # 成对距离与射程
    # ------------------------------
    def pairwise_distance(self, interceptors: List[Dict], targets: List[Dict]) -> np.ndarray:
        print(f"DEBUG: pairwise_distance called. use_cuda={self.use_cuda}", flush=True)
        ni = len(interceptors)
        nt = len(targets)
        if self.use_cuda:
            try:
                # Optimization: Convert to numpy array first to avoid PyTorch UserWarning about slow list-to-tensor conversion
                ip_np = np.array([interceptors[i].get('position', [0, 0, 0]) for i in range(ni)])
                tp_np = np.array([targets[j].get('position', [0, 0, 0]) for j in range(nt)])
                
                print("DEBUG: pairwise_distance creating tensors", flush=True)
                ip = torch.tensor(ip_np, dtype=torch.float32, device='cuda')
                tp = torch.tensor(tp_np, dtype=torch.float32, device='cuda')
                ip = ip.unsqueeze(1).expand(ni, nt, 3)
                tp = tp.unsqueeze(0).expand(ni, nt, 3)
                d = torch.linalg.norm(tp - ip, dim=2)
                print("DEBUG: pairwise_distance cuda done", flush=True)
                return d.detach().cpu().numpy()
            except Exception as e:
                print(f"DEBUG: pairwise_distance CUDA failed: {e}. Fallback to CPU.", flush=True)
                # Fallback to CPU if CUDA fails
                pass

        dmat = np.zeros((ni, nt), dtype=float)
        for i in range(ni):
            ip = np.array(interceptors[i].get('position', [0, 0, 0]), dtype=float)
            for j in range(nt):
                tp = np.array(targets[j].get('position', [0, 0, 0]), dtype=float)
                dmat[i, j] = float(np.linalg.norm(ip - tp))
        return dmat

    def in_range(self, interceptor: Dict, target: Dict) -> bool:
        max_range_km = float(self.weapon_specs.get('max_range_km', 100.0))
        max_range = max_range_km * 1000.0
        ip = np.array(interceptor.get('position', [0, 0, 0]), dtype=float)
        tp = np.array(target.get('position', [0, 0, 0]), dtype=float)
        dist = float(np.linalg.norm(ip - tp))
        return bool(dist <= max_range)

    # ------------------------------
    # TTI 数值估计（非直线目标 + 直线拦截器）
    # ------------------------------
    def estimate_intercept_time(self, interceptor: Dict, target: Dict) -> float:
        if getattr(self, 'fast_mode', False):
            ip0 = np.array(interceptor.get('position', [0, 0, 0]), dtype=float)
            tp0 = np.array(target.get('position', [0, 0, 0]), dtype=float)
            vi = float(interceptor.get('speed', float(self.weapon_specs.get('speed_mps', 300.0))))
            intercept_radius = float(self.weapon_specs.get('intercept_radius_m', 50.0))
            max_flight_time = float(self.weapon_specs.get('max_flight_time_s', 120.0))
            current_flight_time = float(interceptor.get('flight_time', 0.0))
            remaining_time = max(0.0, max_flight_time - current_flight_time)
            
            dist = float(np.linalg.norm(tp0 - ip0))
            if vi <= 1e-9 or remaining_time <= 1e-9:
                return float('inf')
            t_est = max(0.0, (dist - intercept_radius) / vi)
            if t_est > remaining_time:
                return float('inf')
                
            t_defense = self.time_to_defended_zone(target)
            t_max = min(self.max_time_solver, t_defense if np.isfinite(t_defense) else self.max_time_solver)
            return t_est if t_est <= t_max else float('inf')
            
        # 支持蒙特卡洛 TTI 估计模式
        if getattr(self, 'tti_mode', 'deterministic') == 'monte_carlo':
            return self._estimate_intercept_time_monte_carlo(interceptor, target)
            
        # 初始化状态
        ip0 = np.array(interceptor.get('position', [0, 0, 0]), dtype=float)
        speed = float(interceptor.get('speed', float(self.weapon_specs.get('speed_mps', 300.0))))
        intercept_radius = float(self.weapon_specs.get('intercept_radius_m', 50.0))
        max_flight_time = float(self.weapon_specs.get('max_flight_time_s', 120.0))
        current_flight_time = float(interceptor.get('flight_time', 0.0))
        remaining_time = max(0.0, max_flight_time - current_flight_time)
        
        if speed <= 1e-9 or remaining_time <= 1e-9:
            return float('inf')
            
        # 初始速度向量
        vel_m = np.array(interceptor.get('velocity', [0,0,0]), dtype=float)
        if np.linalg.norm(vel_m) < 1e-3:
            # 未发射，假设初始朝向目标当前位置
            tp0 = np.array(target.get('position', [0, 0, 0]), dtype=float)
            d0 = tp0 - ip0
            d0_norm = np.linalg.norm(d0)
            if d0_norm < 1e-3:
                return 0.0 # 就在脸上
            vel_m = d0 / d0_norm * speed
            
        sim_pos_m = ip0.copy()
        sim_vel_m = vel_m.copy()
        
        # 模拟参数
        t_max = min(self.max_time_solver, remaining_time)
        dt = self.dt_solver
        t = 0.0
        
        # 目标防御区时间限制
        t_defense = self.time_to_defended_zone(target)
        if np.isfinite(t_defense):
            t_max = min(t_max, t_defense)
            
        while t <= t_max:
            # 获取目标当前时刻状态预测
            pos_t, vel_t = self._predict_future_with_ekf_or_model(target, tau=t)
            
            # 判定距离
            dist = float(np.linalg.norm(pos_t - sim_pos_m))
            if dist <= intercept_radius:
                return t
                
            # 步进拦截器 (PN 制导)
            sim_pos_m, sim_vel_m = self._step_interceptor_dynamics_pure(
                sim_pos_m, sim_vel_m, pos_t, vel_t, dt, speed
            )
            
            t += dt
            
        return float('inf')

    def _estimate_intercept_time_monte_carlo(self, interceptor: Dict, target: Dict) -> float:
        """蒙特卡洛 TTI 估计：在未来预测上叠加高斯扰动，统计若干样本的拦截时间均值。"""
        # 初始化状态
        ip0 = np.array(interceptor.get('position', [0, 0, 0]), dtype=float)
        speed = float(interceptor.get('speed', float(self.weapon_specs.get('speed_mps', 300.0))))
        intercept_radius = float(self.weapon_specs.get('intercept_radius_m', 50.0))
        max_flight_time = float(self.weapon_specs.get('max_flight_time_s', 120.0))
        current_flight_time = float(interceptor.get('flight_time', 0.0))
        remaining_time = max(0.0, max_flight_time - current_flight_time)
        
        if speed <= 1e-9 or remaining_time <= 1e-9:
            return float('inf')
            
        # 初始速度向量
        vel_m_init = np.array(interceptor.get('velocity', [0,0,0]), dtype=float)
        if np.linalg.norm(vel_m_init) < 1e-3:
            tp0 = np.array(target.get('position', [0, 0, 0]), dtype=float)
            d0 = tp0 - ip0
            d0_norm = np.linalg.norm(d0)
            if d0_norm < 1e-3:
                return 0.0
            vel_m_init = d0 / d0_norm * speed
            
        t_defense = self.time_to_defended_zone(target)
        t_max = min(self.max_time_solver, remaining_time)
        if np.isfinite(t_defense):
            t_max = min(t_max, t_defense)
            
        dt = self.dt_solver

        n = max(1, int(getattr(self, 'mc_samples', 20)))
        noise_std = float(getattr(self, 'mc_noise_std', 50.0))
        
        sum_tti = 0.0
        valid_samples = 0
        
        for k in range(n):
            sim_pos_m = ip0.copy()
            sim_vel_m = vel_m_init.copy()
            t = 0.0
            found = False
            
            while t <= t_max:
                pos_t, vel_t = self._predict_future_with_ekf_or_model(target, tau=t)
                
                # Add noise
                if noise_std > 1e-9:
                    noise = self.rng.normal(0.0, noise_std, size=3)
                    pos_t = pos_t + noise
                
                dist = float(np.linalg.norm(pos_t - sim_pos_m))
                if dist <= intercept_radius:
                    sum_tti += t
                    valid_samples += 1
                    found = True
                    break
                    
                sim_pos_m, sim_vel_m = self._step_interceptor_dynamics_pure(
                    sim_pos_m, sim_vel_m, pos_t, vel_t, dt, speed
                )
                t += dt
            
            if not found:
                pass
 
        if valid_samples == 0:
            return float('inf')
        return sum_tti / valid_samples

    def pairwise_tti(self, interceptors: List[Dict], targets: List[Dict]) -> np.ndarray:
        ni = len(interceptors)
        nt = len(targets)
        tti = np.zeros((ni, nt), dtype=float)
        if getattr(self, 'fast_mode', False):
            ir = float(self.weapon_specs.get('intercept_radius_m', 50.0))
            if self.use_cuda:
                try:
                    # Optimization: Convert to numpy array first
                    ip_np = np.array([interceptors[i].get('position', [0, 0, 0]) for i in range(ni)])
                    tp_np = np.array([targets[j].get('position', [0, 0, 0]) for j in range(nt)])
                    vi_np = np.array([float(interceptors[i].get('speed', float(self.weapon_specs.get('speed_mps', 300.0)))) for i in range(ni)])

                    ip = torch.tensor(ip_np, dtype=torch.float32, device='cuda')
                    tp = torch.tensor(tp_np, dtype=torch.float32, device='cuda')
                    vi = torch.tensor(vi_np, dtype=torch.float32, device='cuda')
                    ip_e = ip.unsqueeze(1).expand(ni, nt, 3)
                    tp_e = tp.unsqueeze(0).expand(ni, nt, 3)
                    d = torch.linalg.norm(tp_e - ip_e, dim=2)
                    vi_e = vi.unsqueeze(1).expand(ni, nt)
                    t_est = torch.clamp((d - ir) / torch.clamp(vi_e, min=1e-9), min=0.0)
                    return t_est.detach().cpu().numpy()
                except RuntimeError:
                    # Fallback to CPU
                    pass

            vi_arr = np.array([float(interceptors[i].get('speed', float(self.weapon_specs.get('speed_mps', 300.0)))) for i in range(ni)], dtype=float)
            ip = np.array([interceptors[i].get('position', [0, 0, 0]) for i in range(ni)], dtype=float)
            tp = np.array([targets[j].get('position', [0, 0, 0]) for j in range(nt)], dtype=float)
            for i in range(ni):
                for j in range(nt):
                    d = float(np.linalg.norm(tp[j] - ip[i]))
                    vi = max(1e-9, vi_arr[i])
                    t_est = max(0.0, (d - ir) / vi)
                    tti[i, j] = t_est
            return tti
        for i in range(ni):
            for j in range(nt):
                tti[i, j] = self.estimate_intercept_time(interceptors[i], targets[j])
        return tti

    # ------------------------------
    # 目标到防御区时间（数值）
    # ------------------------------
    def time_to_defended_zone(self, target: Dict) -> float:
        center = self._get_defense_center(target)
        radius_m = float(self.defended_zone.get('radius_m', 0.0))
        if radius_m <= 1e-6:
            return float('inf')

        # 若已在防御区内
        pos0 = np.array(target.get('position', [0, 0, 0]), dtype=float)
        if np.linalg.norm(pos0 - center) <= radius_m:
            return 0.0

        # 数值推进直到进入防御区或达到最大时间（使用 EKF 或模型预测）
        t = 0.0
        while t <= self.max_time_solver:
            pos_pred, _ = self._predict_future_with_ekf_or_model(target, tau=t)
            if np.linalg.norm(pos_pred - center) <= radius_m:
                return t
            t += self.dt_solver
        return float('inf')

    # ------------------------------
    # 内部：目标状态推进（一步）
    # ------------------------------
    def _advance_target_state(self, t: Dict, dt: float):
        pos = np.array(t.get('position', [0, 0, 0]), dtype=float)
        vel = np.array(t.get('velocity', [0, 0, 0]), dtype=float)
        pos_next, vel_next = self._predict_target_next(pos, vel, dt, t, step_index=0)
        t['position'] = pos_next
        t['velocity'] = vel_next

    def _predict_target_next(self, pos: np.ndarray, vel: np.ndarray, dt: float, target: Dict, step_index: int):
        ttype = str(target.get('type', 'cruise'))

        # 通用速度抖动
        def jitter(v: np.ndarray, std: float) -> np.ndarray:
            if std <= 1e-9:
                return v
            return v + self.rng.normal(0.0, std, size=v.shape)

        if ttype == 'ballistic':
            params = self.motion_specs.get('ballistic', {})
            g = float(params.get('gravity_mps2', 9.81))
            scatter_sigma = float(params.get('scatter_sigma_m', 0.0))
            a = np.array([0.0, 0.0, -g], dtype=float)
            # 围绕已知落点存在散布（缓慢漂移）：
            impact = np.array(target.get('impact_point', pos), dtype=float)
            to_impact = impact - pos
            drift = np.zeros(3, dtype=float)
            if scatter_sigma > 0:
                drift = self.rng.normal(0.0, scatter_sigma, size=3) * 0.01  # 小幅偏移
            # 下落段：主要 z 方向受重力影响
            vel_next = vel + a * dt
            vel_next = jitter(vel_next, float(self.motion_specs.get('ballistic', {}).get('speed_jitter_std', 0.0)))
            pos_next = pos + vel_next * dt + drift
            return pos_next, vel_next

        if ttype == 'cruise':
            params = self.motion_specs.get('cruise', {})
            amp = float(params.get('lateral_amp_m', 20.0))
            omega = float(params.get('lateral_omega', 0.01))
            dive_threshold = float(params.get('dive_threshold_m', 3000.0))
            dive_rate = float(params.get('dive_rate_mps', 100.0))
            dive_alt_th = float(params.get('dive_altitude_threshold_m', 1000.0))
            trigger_mode = str(params.get('dive_trigger_mode', 'distance_xyz'))
            impact = np.array(target.get('impact_point', pos), dtype=float)
            dist_to_impact_xyz = float(np.linalg.norm(impact - pos))
            dist_to_impact_xy = float(np.linalg.norm(impact[:2] - pos[:2]))
            # 判定是否继续平飞（横向扰动）
            def should_cruise() -> bool:
                if trigger_mode == 'distance_xy':
                    return dist_to_impact_xy > dive_threshold
                if trigger_mode == 'altitude':
                    return float(pos[2]) > dive_alt_th
                # 默认 distance_xyz
                return dist_to_impact_xyz > dive_threshold

            # 平飞阶段：横向扰动，速度模长基本不变
            if should_cruise():
                if np.linalg.norm(vel[:2]) > 1e-6:
                    v2 = vel[:2]
                    ortho = np.array([-v2[1], v2[0]], dtype=float)
                    ortho_norm = np.linalg.norm(ortho)
                    if ortho_norm > 1e-9:
                        ortho = ortho / ortho_norm
                    lateral_2d = amp * np.sin(omega * (step_index + 1) * dt) * ortho
                    lateral = np.array([lateral_2d[0], lateral_2d[1], 0.0], dtype=float)
                else:
                    lateral = np.array([amp * np.sin(omega * (step_index + 1) * dt), 0.0, 0.0], dtype=float)
                vel_next = jitter(vel, float(params.get('speed_jitter_std', 0.0)))
                pos_next = pos + vel_next * dt + lateral
            else:
                # 俯冲阶段：增加负的 z 分量
                vel_next = vel.copy()
                vel_next[2] = vel_next[2] - dive_rate
                vel_next = jitter(vel_next, float(params.get('speed_jitter_std', 0.0)))
                pos_next = pos + vel_next * dt
            return pos_next, vel_next

        # aircraft：水平转向，角速度恒定（绕 z 轴）
        params = self.motion_specs.get('aircraft', {})
        w = float(params.get('turn_rate_rad_s', 0.005))
        vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
        cosw = np.cos(w * dt)
        sinw = np.sin(w * dt)
        vx2 = vx * cosw - vy * sinw
        vy2 = vx * sinw + vy * cosw
        vel_next = np.array([vx2, vy2, vz], dtype=float)
        vel_next = jitter(vel_next, float(params.get('speed_jitter_std', 0.0)))
        pos_next = pos + vel_next * dt
        return pos_next, vel_next

    # ------------------------------
    # 防御区中心选择
    # ------------------------------
    def _get_defense_center(self, target: Optional[Dict] = None) -> np.ndarray:
        """根据 defended_zone.center_mode 返回防御区中心。
        - fixed：使用 defended_zone.center；若未配置则使用 defense_points[0]；否则原点。
        - nearest：若存在 defense_points，选择距离 target 当前坐标最近的点；若无 target 或列表为空，回退到 fixed。
        """
        mode = str(self.defended_zone.get('center_mode', 'fixed'))
        if mode == 'nearest' and self.defense_points and target is not None:
            try:
                tpos = np.array(target.get('position', [0, 0, 0]), dtype=float)
                best = None
                best_d = float('inf')
                for pt in self.defense_points:
                    p = np.array(pt.get('position', [0, 0, 0]), dtype=float)
                    d = float(np.linalg.norm(tpos - p))
                    if d < best_d:
                        best_d = d
                        best = p
                if best is not None:
                    return best
            except Exception:
                pass
        # fixed 模式或无法选择最近时：
        if 'center' in self.defended_zone:
            return np.array(self.defended_zone.get('center', [0, 0, 0]), dtype=float)
        if self.defense_points:
            return np.array(self.defense_points[0]['position'], dtype=float)
        return np.array([0.0, 0.0, 0.0], dtype=float)

    # ------------------------------
    # EKF 接口：未来预测或模型模拟
    # ------------------------------
    def _predict_future_with_ekf_or_model(self, target: Dict, tau: float):
        """优先调用 EKFTracker.predict_horizon，否则用模型重复推进 tau。"""
        try:
            from src.utils.filter import EKFTracker  # 接口占位，具体实现由 utils/filter.py 提供
        except Exception:
            EKFTracker = None

        pos0 = np.array(target.get('position', [0, 0, 0]), dtype=float)
        vel0 = np.array(target.get('velocity', [0, 0, 0]), dtype=float)

        tracker = target.get('tracker', None)
        if tracker is not None and hasattr(tracker, 'predict_horizon'):
            try:
                pred = tracker.predict_horizon(pos0, vel0, tau, target)
                p = np.array(pred.get('position', pos0), dtype=float)
                v = np.array(pred.get('velocity', vel0), dtype=float)
                return p, v
            except Exception:
                # 若滤波器异常，回退到模型模拟
                pass

        # 模型模拟：重复调用 _predict_target_next 直到 tau
        if tau <= 0:
            return pos0, vel0
        steps = max(1, int(np.ceil(tau / self.dt_solver)))
        dt = float(tau / steps)
        pos, vel = pos0.copy(), vel0.copy()
        for k in range(steps):
            pos, vel = self._predict_target_next(pos, vel, dt, target, step_index=k)
        return pos, vel
