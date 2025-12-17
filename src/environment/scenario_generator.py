"""
Scenario Generator
------------------
为环境提供可复现的目标与拦截器初始化，并与 PhysicsEngine / EKFTracker 接口对齐。

核心功能：
- load_config: 从 YAML 路径加载配置字典；
- generate_random_targets: 生成包含 type/position/velocity/impact_point/motion_specs/tracker 的目标列表；
- generate_random_interceptors: 生成拦截器列表（位置/速度/弹药/命中概率）；
- make_scenario: 一次性构造场景（目标 + 拦截器），便于上层环境使用。

与 PhysicsEngine/EKFTracker 的接口对齐要点：
- 目标字典包含：
  - position: np.ndarray(3,)
  - velocity: np.ndarray(3,)
  - type: 'ballistic' | 'cruise' | 'aircraft'
  - impact_point: np.ndarray(3,)（用于弹道/巡航）
  - motion_specs: Dict（三类目标参数，键与 physics_engine 中一致）
  - tracker: EKFTracker 实例（可选，但推荐附加以获得更稳健的未来预测）

默认值策略：
- defended_zone.center 默认为原点；radius_m=5000；center_mode='fixed'；
- motion_specs 与 tti_solver 若未配置则提供合理默认值；
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import yaml

try:
    # 可选依赖：用于目标未来预测（physics_engine 会优先调用）
    from src.utils.filter import EKFTracker
except Exception:
    EKFTracker = None  # 允许在未实现时仍可生成场景


def load_config(path: Optional[str]) -> Dict:
    """从 YAML 路径加载配置；若路径为空或加载失败，返回空字典。"""
    if not path:
        return {}
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
            return cfg
    except Exception:
        return {}


def _ensure_defaults(env_cfg: Dict) -> Dict:
    """补齐 env_cfg 中的关键默认值，避免上层使用时出现缺失。"""
    cfg = dict(env_cfg or {})

    # 防御区默认值
    dz = cfg.get('defended_zone', {})
    dz.setdefault('center', [0.0, 0.0, 0.0])
    dz.setdefault('radius_m', 5000.0)
    dz.setdefault('center_mode', 'fixed')
    cfg['defended_zone'] = dz

    # 武器参数默认值
    ws = cfg.get('weapon_specs', {})
    ws.setdefault('max_range_km', 100.0)
    ws.setdefault('speed_mps', 300.0)
    ws.setdefault('intercept_radius_m', 50.0)
    ws.setdefault('ammo_per_unit', 4)
    ws.setdefault('base_hit_prob', 0.7)
    cfg['weapon_specs'] = ws

    # 目标运动参数默认值（与 physics_engine 对齐）
    ms = cfg.get('motion_specs', {})
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
    cfg['motion_specs'] = ms

    # TTI 求解器默认值（与 physics_engine 对齐）
    ts = cfg.get('tti_solver', {})
    ts.setdefault('dt', 0.5)
    ts.setdefault('max_time', 600.0)
    ts.setdefault('mode', 'deterministic')
    ts.setdefault('mc_samples', 20)
    ts.setdefault('mc_noise_std', 50.0)
    ts.setdefault('aim_mode', 'predicted_future')
    cfg['tti_solver'] = ts

    # 随机种子
    cfg.setdefault('seed', 42)

    # 场景规模默认值
    cfg.setdefault('max_targets', 20)
    cfg.setdefault('max_interceptors', 10)
    cfg.setdefault('time_step', 1.0)
    cfg.setdefault('horizon', 200)
    return cfg


def _rng_from_seed(seed: Optional[int]) -> np.random.RandomState:
    return np.random.RandomState(int(seed) if seed is not None else 42)


def _sample_types(num_targets: int, env_cfg: Dict, rng: np.random.RandomState) -> List[str]:
    dist = env_cfg.get('target_type_distribution', {
        'ballistic': 1/3, 'cruise': 1/3, 'aircraft': 1/3,
    })
    names = ['ballistic', 'cruise', 'aircraft']
    probs = np.array([float(dist.get(k, 0.0)) for k in names], dtype=float)
    if probs.sum() <= 0:
        probs = np.array([1/3, 1/3, 1/3], dtype=float)
    probs = probs / probs.sum()
    return list(rng.choice(names, size=num_targets, p=probs))


def _sample_impact_point(defended_zone: Dict, rng: np.random.RandomState, spread_m: float = 1000.0) -> np.ndarray:
    """围绕防御区中心采样一个 impact_point（高斯散布）。"""
    center = np.asarray(defended_zone.get('center', [0.0, 0.0, 0.0]), dtype=float)
    jitter = rng.normal(0.0, spread_m, size=3)
    # 保持落点略高于地面（z >= 0）
    impact = center + jitter
    impact[2] = max(0.0, impact[2])
    return impact


def generate_random_targets(num_targets: int = 50,
                            env_cfg: Optional[Dict] = None,
                            seed: Optional[int] = None) -> List[Dict]:
    """生成随机目标集合，字段与 PhysicsEngine/EKFTracker 对齐。

    返回的每个目标字典包含：
    - position, velocity, type, impact_point, motion_specs, value, threat, alive, tracker(可选)
    """
    cfg = _ensure_defaults(env_cfg or {})
    rng = _rng_from_seed(seed if seed is not None else cfg.get('seed', 42))
    types = _sample_types(num_targets, cfg, rng)

    dz = cfg.get('defended_zone', {})
    mspec = cfg.get('motion_specs', {})

    # EKF tracker 参数（若可用则附加）
    tracker_dt = float(cfg.get('tti_solver', {}).get('dt', 0.5))
    tracker_q_pos = 1.0
    tracker_q_vel = 0.5
    tracker_use_model = True

    targets: List[Dict] = []
    for idx, ttype in enumerate(types):
        # 初始位置：在较大方形区域内均匀采样
        pos = rng.uniform(-10000, 10000, size=3)
        # 初始速度：按类型采样更贴近的初值
        if ttype == 'ballistic':
            # 向落点方向有一定水平速度，竖直速度为负（下降）
            vxy = rng.uniform(-200, 200, size=2)
            vz = -rng.uniform(100, 300)
            vel = np.array([vxy[0], vxy[1], vz], dtype=float)
        elif ttype == 'cruise':
            # 水平速度主导，竖直速度较小
            vxy = rng.uniform(-150, 150, size=2)
            vz = rng.uniform(-10, 20)
            vel = np.array([vxy[0], vxy[1], vz], dtype=float)
        else:  # aircraft
            # 水平速度为主，速度模长适中
            speed = rng.uniform(80, 200)
            angle = rng.uniform(-np.pi, np.pi)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            vz = rng.uniform(-5, 5)
            vel = np.array([vx, vy, vz], dtype=float)

        impact = _sample_impact_point(dz, rng, spread_m=1500.0)

        tgt = {
            'id': idx,
            'type': ttype,
            'position': np.asarray(pos, dtype=float),
            'velocity': np.asarray(vel, dtype=float),
            'impact_point': np.asarray(impact, dtype=float),
            'motion_specs': mspec,
            'value': float(rng.uniform(0.5, 1.0)),
            'threat': float(rng.uniform(0.5, 1.0)),
            'alive': True,
        }

        # 可选：附加 EKFTracker 实例，增强未来预测
        if EKFTracker is not None:
            try:
                tgt['tracker'] = EKFTracker(dt=tracker_dt, q_pos_std=tracker_q_pos, q_vel_std=tracker_q_vel,
                                            use_model_aided=tracker_use_model, seed=cfg.get('seed', 42))
            except Exception:
                pass

        targets.append(tgt)

    return targets


def generate_random_interceptors(num_interceptors: int = 10,
                                 env_cfg: Optional[Dict] = None,
                                 seed: Optional[int] = None) -> List[Dict]:
    """生成随机拦截器集合，字段与 WTAEnv/WTAEngine 对齐：position/speed/ammo/hit_prob。"""
    cfg = _ensure_defaults(env_cfg or {})
    rng = _rng_from_seed(seed if seed is not None else cfg.get('seed', 42))
    ws = cfg.get('weapon_specs', {})

    items: List[Dict] = []
    for _ in range(num_interceptors):
        it = {
            'position': rng.uniform(-10000, 10000, size=3),
            'speed': float(ws.get('speed_mps', 300.0)),
            'ammo': int(ws.get('ammo_per_unit', 4)),
            'hit_prob': float(ws.get('base_hit_prob', 0.7)),
        }
        items.append(it)
    return items


def make_scenario(env_cfg: Optional[Dict] = None,
                  num_targets: Optional[int] = None,
                  num_interceptors: Optional[int] = None,
                  seed: Optional[int] = None) -> Dict:
    """构造完整场景字典：{'targets': [...], 'interceptors': [...], 'env_cfg': cfg}。

    - 若 num_targets/num_interceptors 未提供，则使用 env_cfg 中的 max_targets/max_interceptors；
    - 返回的 env_cfg 已补齐默认键，可直接传给 PhysicsEngine/WTAEnv。
    """
    cfg = _ensure_defaults(env_cfg or {})
    nt = int(num_targets if num_targets is not None else cfg.get('max_targets', 20))
    ni = int(num_interceptors if num_interceptors is not None else cfg.get('max_interceptors', 10))

    tgts = generate_random_targets(nt, cfg, seed)
    itcs = generate_random_interceptors(ni, cfg, seed)
    return {
        'targets': tgts,
        'interceptors': itcs,
        'env_cfg': cfg,
    }


__all__ = ['load_config', 'generate_random_targets', 'generate_random_interceptors', 'make_scenario']