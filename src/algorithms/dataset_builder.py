import os
import json
import math
import copy
from typing import Dict, Any, List

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger
from src.environment.wta_env import WTAEnv
from src.models.hdt_iql import HDTIQLPolicy


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, 'r', encoding='UTF-8') as f:
        return yaml.safe_load(f) or {}


def _prepare_env_cfg(env_cfg: Dict[str, Any], scenario_cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """根据场景配置对 env_cfg 进行轻量覆盖：seed、TTI 上限、运动扰动等。"""
    cfg = copy.deepcopy(env_cfg or {})
    # 随机种子
    cfg['seed'] = int(seed)
    # TTI 上限
    tti_max = scenario_cfg.get('time_to_impact')
    if tti_max is not None:
        ts = cfg.get('tti_solver', {})
        ts['max_time'] = float(tti_max)
        cfg['tti_solver'] = ts
    # 目标散布率（简化映射到运动参数抖动）
    disp = scenario_cfg.get('dispersion_rate')
    if disp is not None:
        ms = cfg.get('motion_specs', {})
        b = ms.get('ballistic', {})
        c = ms.get('cruise', {})
        a = ms.get('aircraft', {})
        # 将 dispersion_rate 映射为速度/位置噪声幅度的缩放（经验映射）
        scale = float(disp)
        b['scatter_sigma_m'] = float(50.0 * scale)
        c['lateral_amp_m'] = float(max(5.0, 20.0 * scale))
        a['speed_jitter_std'] = float(max(0.0, 0.5 * scale))
        ms['ballistic'] = b
        ms['cruise'] = c
        ms['aircraft'] = a
        cfg['motion_specs'] = ms
    return cfg


def _ensure_dirs(root: str) -> Dict[str, str]:
    os.makedirs(root, exist_ok=True)
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    return {'train': train_dir, 'val': val_dir}


def _save_episode(ep: Dict[str, Any], out_path: str, fmt: str, logger):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if fmt == 'pt' and TORCH_AVAILABLE:
        torch.save(ep, out_path)
    elif fmt == 'npz':
        import numpy as np
        # 将字典结构扁平化存储：仅保存基本字段；复杂嵌套采用 JSON 辅助
        steps = ep.get('steps', [])
        rewards = np.array([s.get('reward', 0.0) for s in steps], dtype=float)
        dones = np.array([bool(s.get('done', False)) for s in steps], dtype=bool)
        actions_json = json.dumps([s.get('action', []) for s in steps])
        infos_json = json.dumps([s.get('info', {}) for s in steps])
        # 观测与掩码可能较大，使用 JSON 序列化再由下游解析（保持可读性）
        obs_json = json.dumps([_to_list(s.get('obs', {})) for s in steps])
        masks_json = json.dumps([_to_list(s.get('masks', {})) for s in steps])
        np.savez(out_path, rewards=rewards, dones=dones, actions_json=actions_json,
                 infos_json=infos_json, obs_json=obs_json, masks_json=masks_json,
                 episode_id=ep.get('episode_id'), seed=ep.get('seed'))
    else:
        # 回退到 pickle
        import pickle
        if not out_path.endswith('.pkl'):
            out_path = out_path + '.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump(ep, f)
        logger.warning(f"torch 不可用或不支持的格式 '{fmt}'；已回退到 pickle: {out_path}")


def _to_list(obj: Any) -> Any:
    """将 numpy/tensor 等对象转换为可 JSON 序列的基本 Python 类型。"""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    # dict/list 递归转换
    if isinstance(obj, dict):
        return {k: _to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_list(x) for x in obj]
    # torch.Tensor
    if TORCH_AVAILABLE:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    return obj


def build_offline_dataset(scenario_cfg: Dict[str, Any], data_cfg: Dict[str, Any], output_root: str):
    """生成离线数据集：按 data.yaml 配置运行若干 episode，并序列化到磁盘。

    输出结构：
    - output_root/
      - train/
        - ep_000000.pt 或 .npz/.pkl
        - ...
      - val/
        - ep_000100.pt 或 .npz/.pkl
      - index.json  # 数据集摘要与路径清单
    """
    project_root = _project_root()
    log_dir = os.path.join(project_root, 'experiments', 'results', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger('dataset_builder', os.path.join(log_dir, 'dataset_builder.log'))
    logger.info('开始数据集生成流程')

    # 加载 env/reward/model 配置
    cfg_env = _load_yaml(os.path.join(project_root, 'config', 'env.yaml'))
    cfg_reward = _load_yaml(os.path.join(project_root, 'config', 'reward.yaml'))
    cfg_model = _load_yaml(os.path.join(project_root, 'config', 'model.yaml'))

    # 基本数据参数
    offline = data_cfg.get('offline_dataset', {})
    fmt = str(offline.get('format', 'pt')).lower()
    num_episodes = int(offline.get('num_episodes', 100))
    max_len = int(offline.get('max_len', 256))
    dirs = _ensure_dirs(output_root)
    split_cfg = data_cfg.get('split', {'train': 0.8, 'val': 0.2})
    train_ratio = float(split_cfg.get('train', 0.8))
    train_count = int(math.floor(num_episodes * train_ratio))

    # 场景 seeds
    seeds = list(scenario_cfg.get('seeds', [42]))
    if not seeds:
        seeds = [42]

    # 策略（优先使用 IQL 启发式；DT 在无 torch 时自动降级为 None）
    policy = HDTIQLPolicy(cfg_model)

    index_manifest: Dict[str, Any] = {
        'format': fmt,
        'episodes': [],
        'stats': {
            'num_episodes': num_episodes,
            'train_count': train_count,
            'val_count': num_episodes - train_count,
            'total_steps': 0,
        }
    }

    # 逐个 episode 生成
    for ep_idx in range(num_episodes):
        seed = int(seeds[ep_idx % len(seeds)])
        # 依据场景配置调整 env_cfg
        env_cfg = _prepare_env_cfg(cfg_env, scenario_cfg, seed)
        env = WTAEnv(env_cfg, cfg_reward, cfg_model)
        logger.info(f"开始生成 episode {ep_idx+1}/{num_episodes} | seed={seed}")
        obs = env.reset()

        steps: List[Dict[str, Any]] = []
        done = False
        step_idx = 0
        while not done and step_idx < max_len:
            masks = env.get_action_masks()
            action = _mixed_expert_action(env, obs, masks, env.rng)
            obs, reward, done, info = env.step(action)
            steps.append({
                'obs': obs,
                'masks': masks,
                'action': action,
                'reward': float(reward),
                'done': bool(done),
                'info': info,
            })
            step_idx += 1

        ep_id = f"ep_{ep_idx:06d}"
        split = 'train' if ep_idx < train_count else 'val'
        out_name = f"{ep_id}.{fmt if TORCH_AVAILABLE or fmt=='npz' else 'pkl'}"
        out_path = os.path.join(dirs[split], out_name)
        episode_obj = {
            'episode_id': ep_id,
            'seed': seed,
            'steps': steps,
            'env_cfg': env_cfg,
        }
        _save_episode(episode_obj, out_path, fmt, logger)
        index_manifest['episodes'].append({'id': ep_id, 'split': split, 'path': os.path.relpath(out_path, output_root), 'steps': len(steps)})
        index_manifest['stats']['total_steps'] += len(steps)

        logger.info(f"完成 episode {ep_idx+1}/{num_episodes} | split='{split}' | steps={len(steps)} | 输出: {out_path}")

    # 写入索引文件
    with open(os.path.join(output_root, 'index.json'), 'w', encoding='UTF-8') as f:
        json.dump(index_manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"数据集生成完成：episodes={num_episodes}, total_steps={index_manifest['stats']['total_steps']}, 格式='{fmt}'")


def _mixed_expert_action(env, obs, masks, rng):
    assign = masks.get('assign_mask')
    alive = masks.get('alive_mask')
    range_m = masks.get('range_mask')
    capacity = masks.get('capacity_mask')
    defense = masks.get('defense_time_mask')
    ammo = masks.get('ammo_mask')
    base = assign if assign is not None else (range_m or alive or capacity or defense or ammo)
    if base is None:
        return []
    if alive is None: alive = base
    if range_m is None: range_m = base
    if capacity is None: capacity = base
    if defense is None: defense = base
    if ammo is None: ammo = base
    combined = assign & alive & range_m & capacity & defense & ammo
    import numpy as np
    idxs = np.argwhere(combined)
    if idxs.size == 0:
        return []
    tti = masks.get('pairwise_tti')
    if isinstance(tti, np.ndarray) and tti.shape == combined.shape:
        # experts: shortest TTI, highest value, prefer larger defense_time_gap
        vmap = {'ballistic': 3.0, 'cruise': 2.0, 'aircraft': 1.0}
        values = []
        for j, tgt in enumerate(env.state['targets']):
            values.append(vmap.get(str(tgt.get('type', 'cruise')), 1.0))
        def_time = np.array([env.engine.time_to_defended_zone(t) for t in env.state['targets']], dtype=float)
        # random
        if rng.rand() < 0.05:
            pick = idxs[rng.randint(0, idxs.shape[0])]
            return [(int(pick[0]), int(pick[1]))]
        # shortest TTI
        idxs_sorted_tti = sorted(idxs.tolist(), key=lambda ij: tti[ij[0], ij[1]])
        best_tti = idxs_sorted_tti[0]
        # highest value then tti
        idxs_sorted_val = sorted(idxs.tolist(), key=lambda ij: (-values[ij[1]], tti[ij[0], ij[1]], -def_time[ij[1]] if np.isfinite(def_time[ij[1]]) else 0.0))
        if rng.rand() < 0.5:
            i, j = best_tti
        else:
            i, j = idxs_sorted_val[0]
        return [(int(i), int(j))]
    pick = idxs[0]
    return [(int(pick[0]), int(pick[1]))]
