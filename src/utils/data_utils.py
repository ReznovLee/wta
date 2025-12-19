import os
import json
from typing import List, Tuple, Dict, Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def _load_index(data_root: str) -> Dict[str, Any]:
    """加载数据集索引文件 index.json。"""
    index_path = os.path.join(data_root, 'index.json')
    if not os.path.exists(index_path):
        return {'episodes': [], 'stats': {}}
    with open(index_path, 'r', encoding='UTF-8') as f:
        return json.load(f)


def load_scenarios(path: str) -> Dict[str, Any]:
    """加载数据根目录下的索引（或直接加载给定 JSON 路径）。

    返回 manifest 字典：{'episodes': [...], 'stats': {...}}。
    """
    if os.path.isdir(path):
        return _load_index(path)
    else:
        with open(path, 'r', encoding='UTF-8') as f:
            return json.load(f)


def split_train_val(data_root: str, split_cfg: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """根据 index.json 的 split 字段，返回训练与验证文件路径列表（相对 data_root）。"""
    manifest = _load_index(data_root)
    train, val = [], []
    for ep in manifest.get('episodes', []):
        rel_path = ep.get('path')
        if not rel_path:
            continue
        full_path = os.path.join(data_root, rel_path)
        if ep.get('split') == 'train':
            train.append(full_path)
        else:
            val.append(full_path)
    return train, val


def _pad_list(lst: List[Any], pad_to: int, pad_val: Any) -> List[Any]:
    out = lst[:pad_to]
    if len(out) < pad_to:
        out = out + [pad_val] * (pad_to - len(out))
    return out


def to_tensor_batch(episodes: List[Dict[str, Any]], max_len: int) -> Dict[str, Any]:
    B = len(episodes)
    states = []
    action_masks = []
    pairwise_tti = []
    returns_to_go = []
    actions = []
    rewards = []
    dones = []
    d_model = 256
    for ep in episodes:
        steps = ep.get('steps', [])
        if steps:
            m0 = steps[0].get('masks', {})
            assign = m0.get('assign_mask')
            if assign is not None:
                num_i = len(assign)
                num_j = len(assign[0]) if len(assign) > 0 else 0
            else:
                num_i, num_j = 0, 0
        else:
            num_i, num_j = 0, 0
        ep_states = []
        ep_masks = []
        ep_tti = []
        ep_actions = []
        ep_rewards = []
        ep_dones = []
        for s in steps[:max_len]:
            obs = s.get('obs', {})
            masks = s.get('masks', {})
            tp = obs.get('targets_pos')
            ip = obs.get('interceptors_pos')
            alive = obs.get('targets_alive')
            tti = obs.get('pairwise_tti')
            dist = obs.get('pairwise_distance')
            val = obs.get('targets_val')
            thr = obs.get('targets_thr')
            ammo_arr = obs.get('interceptors_ammo')
            hp_arr = obs.get('interceptors_hit_prob')
            spd_arr = obs.get('interceptors_speed')
            type_id = obs.get('targets_type_id')
            tcur = obs.get('t', 0.0)
            nt = int(len(tp) if tp is not None else 0)
            ni = int(len(ip) if ip is not None else 0)
            def _stats(a):
                if a is None:
                    return 0.0, 0.0, 0.0
                try:
                    arr = np.asarray(a, dtype=float)
                    if arr.size == 0:
                        return 0.0, 0.0, 0.0
                    return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr))
                except Exception:
                    return 0.0, 0.0, 0.0
            ammo_mean, ammo_min, ammo_max = _stats(ammo_arr)
            hp_mean, hp_min, hp_max = _stats(hp_arr)
            spd_mean, spd_min, spd_max = _stats(spd_arr)
            val_mean, val_min, val_max = _stats(val)
            thr_mean, thr_min, thr_max = _stats(thr)
            tti_mean, tti_min, tti_max = _stats(tti)
            dist_mean, dist_min, dist_max = _stats(dist)
            alive_sum = float(sum(alive) if isinstance(alive, (list, np.ndarray)) and len(alive) > 0 else 0.0)
            type_hist = np.zeros(3, dtype=float)
            if isinstance(type_id, (list, np.ndarray)) and len(type_id) > 0:
                arr = np.asarray(type_id, dtype=int)
                for k in [0, 1, 2]:
                    type_hist[k] = float(np.sum(arr == k))
                if nt > 0:
                    type_hist = type_hist / float(nt)
            # threat potential and resource tightness
            eps = 1e-3
            tp_sum = 0.0
            if tti is not None and isinstance(val, (list, np.ndarray)):
                try:
                    import numpy as np
                    val_arr = np.asarray(val, dtype=float)
                    if isinstance(tti, np.ndarray):
                        for jidx in range(val_arr.shape[0]):
                            mt = float(np.min(tti[:, jidx])) if tti.shape[0] > 0 else 0.0
                            tp_sum += float(val_arr[jidx]) / (max(eps, mt))
                except Exception:
                    tp_sum = 0.0
            ammo_mean2 = ammo_mean
            tightness = float(tp_sum) / max(eps, ammo_mean2 + 1.0)
            feat = [
                float(nt),
                float(ni),
                alive_sum,
                ammo_mean,
                ammo_min,
                ammo_max,
                hp_mean,
                hp_min,
                hp_max,
                spd_mean,
                spd_min,
                spd_max,
                val_mean,
                val_min,
                val_max,
                thr_mean,
                thr_min,
                thr_max,
                float(tcur),
                tti_mean,
                tti_min,
                tti_max,
                dist_mean,
                dist_min,
                dist_max,
                float(tp_sum),
                float(tightness),
            ] + list(type_hist.astype(float))
            if len(feat) < d_model:
                feat = feat + [0.0] * (d_model - len(feat))
            else:
                feat = feat[:d_model]
            ep_states.append(feat)
            def get_m(name):
                return masks.get(name)
            combo = None
            for mn in ['assign_mask', 'ammo_mask', 'range_mask', 'alive_mask', 'capacity_mask', 'defense_time_mask']:
                m = get_m(mn)
                if m is None:
                    continue
                combo = m if combo is None else (combo & m)
            ep_masks.append(combo if combo is not None else [[False]*num_j for _ in range(num_i)])
            tti_mat = obs.get('pairwise_tti')
            if tti_mat is None:
                tti_mat = [[0.0]*num_j for _ in range(num_i)]
            ep_tti.append(tti_mat)
            ep_actions.append(s.get('action', []))
            ep_rewards.append(float(s.get('reward', 0.0)))
            ep_dones.append(bool(s.get('done', False)))
        rtg = []
        acc = 0.0
        for r in reversed(ep_rewards):
            acc = float(r) + acc
            rtg.append(acc)
        rtg = list(reversed(rtg))
        states.append(_pad_list(ep_states, max_len, [0.0] * d_model))
        action_masks.append(_pad_list(ep_masks, max_len, [[False]*num_j for _ in range(num_i)]))
        pairwise_tti.append(_pad_list(ep_tti, max_len, [[0.0]*num_j for _ in range(num_i)]))
        actions.append(_pad_list(ep_actions, max_len, []))
        rewards.append(_pad_list(ep_rewards, max_len, 0.0))
        dones.append(_pad_list(ep_dones, max_len, True))
        returns_to_go.append(_pad_list(rtg, max_len, 0.0))
    if TORCH_AVAILABLE:
        states = torch.tensor(states, dtype=torch.float32)
        try:
            action_masks = torch.tensor(action_masks, dtype=torch.bool)
        except Exception:
            pass
        try:
            pairwise_tti = torch.tensor(pairwise_tti, dtype=torch.float32)
        except Exception:
            pass
        returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
    return {'states': states, 'action_masks': action_masks, 'pairwise_tti': pairwise_tti, 'returns_to_go': returns_to_go, 'actions': actions, 'rewards': rewards, 'dones': dones}
