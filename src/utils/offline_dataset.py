import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    Dataset = object
    DataLoader = None

import numpy as np

from .data_utils import load_scenarios, split_train_val, to_tensor_batch


def _load_episode_file(path: str) -> Dict[str, Any]:
    """加载单个 episode 文件，支持 .pt/.pkl/.npz。

    统一返回结构：{'episode_id': str, 'seed': int, 'steps': List[step_dict]}
    其中 step_dict 至少包含：'obs', 'masks', 'action', 'reward', 'done', 'info'
    """
    if path.endswith('.pt') and TORCH_AVAILABLE:
        obj = torch.load(path)
        return obj
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    if path.endswith('.npz'):
        npz = np.load(path, allow_pickle=True)
        rewards = npz['rewards']
        dones = npz['dones']
        actions_json = json.loads(str(npz['actions_json']))
        infos_json = json.loads(str(npz['infos_json']))
        obs_json = json.loads(str(npz['obs_json']))
        masks_json = json.loads(str(npz['masks_json']))
        ep_id = npz.get('episode_id')
        if hasattr(ep_id, 'item'):
            ep_id = ep_id.item()
        seed = npz.get('seed')
        if hasattr(seed, 'item'):
            seed = int(seed.item())
        steps = []
        T = len(rewards)
        for t in range(T):
            steps.append({
                'obs': obs_json[t],
                'masks': masks_json[t],
                'action': actions_json[t],
                'reward': float(rewards[t]),
                'done': bool(dones[t]),
                'info': infos_json[t],
            })
        return {'episode_id': ep_id, 'seed': seed, 'steps': steps}
    # 尝试通用 JSON
    if path.endswith('.json'):
        with open(path, 'r', encoding='UTF-8') as f:
            return json.load(f)
    # 兜底：返回空结构
    return {'episode_id': os.path.basename(path), 'seed': 0, 'steps': []}


class OfflineEpisodesDataset(Dataset):
    """离线 episodes 数据集：按 index.json 的 split 返回指定集合的 episode。

    - data_root: 数据集根目录（包含 index.json、train/、val/）
    - split: 'train' 或 'val'
    - max_len: 采样或填充的最大序列长度（用于后续 collate）
    """

    def __init__(self, data_root: str, split: str = 'train', max_len: int = 256):
        self.data_root = data_root
        self.split = split
        self.max_len = int(max_len)
        paths_train, paths_val = split_train_val(data_root, {})
        self.paths = paths_train if split == 'train' else paths_val

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        ep = _load_episode_file(path)
        return ep


def collate_episodes(batch: List[Dict[str, Any]], max_len: int) -> Dict[str, Any]:
    """将 episode 列表聚合为批量张量/列表，使用 data_utils.to_tensor_batch。"""
    return to_tensor_batch(batch, max_len=max_len)


def make_dataloader(data_root: str, split: str = 'train', max_len: int = 256, batch_size: int = 8, shuffle: bool = True):
    """创建可迭代的 DataLoader；在无 torch 环境下返回 Python 迭代器。"""
    dataset = OfflineEpisodesDataset(data_root, split=split, max_len=max_len)
    if TORCH_AVAILABLE:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=lambda batch: collate_episodes(batch, max_len))
    else:
        # 简易迭代器：每次返回一个批，使用 collate_episodes 聚合
        def _iter():
            batch: List[Dict[str, Any]] = []
            for ep in dataset:
                batch.append(ep)
                if len(batch) == batch_size:
                    yield collate_episodes(batch, max_len)
                    batch = []
            if batch:
                yield collate_episodes(batch, max_len)
        return _iter()
