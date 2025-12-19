#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
from src.utils.runtime_env import configure_runtime
"""
训练入口：HDT-IQL
加载配置，构建环境/策略/训练器，先进行离线DT预训练，然后进行IQL强化学习。
"""
import yaml
from src.utils.logger import get_logger
from src.algorithms.hdt_iql_trainer import HDTIQLTrainer
from src.environment.wta_env import WTAEnv
from src.models.hdt_iql import HDTIQLPolicy
from src.algorithms.replay_buffer import ReplayBuffer
from src.algorithms.dataset_builder import build_offline_dataset


def load_yaml(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return yaml.safe_load(f)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(root, '..'))
    configure_runtime()
    cfg_env = load_yaml(os.path.join(project_root, 'config', 'env.yaml'))
    cfg_reward = load_yaml(os.path.join(project_root, 'config', 'reward.yaml'))
    cfg_model = load_yaml(os.path.join(project_root, 'config', 'model.yaml'))
    cfg_data = load_yaml(os.path.join(project_root, 'config', 'data.yaml'))
    cfg_exp = load_yaml(os.path.join(project_root, 'config', 'experiment.yaml'))

    log_path = os.path.join(project_root, cfg_exp['paths']['logs'])
    os.makedirs(log_path, exist_ok=True)
    logger = get_logger('train', os.path.join(log_path, 'train.log'))

    # 生成离线数据集（若不存在）
    build_offline_dataset(cfg_data['scenario'], cfg_data, cfg_data['offline_dataset']['root'])

    # 构建环境与策略
    env = WTAEnv(cfg_env, cfg_reward, cfg_model)
    policy = HDTIQLPolicy(cfg_model)
    trainer = HDTIQLTrainer(env, policy, logger, cfg_exp)

    # 离线预训练
    logger.info('开始 Decision Transformer 的离线预训练...')
    trainer.pretrain_dt(cfg_data)

    # IQL 强化学习阶段
    logger.info('开始 IQL 强化学习阶段...')
    buffer = ReplayBuffer(max_size=100000, seq_len=cfg_data['offline_dataset']['max_len'])
    trainer.train_iql(env, buffer, total_steps=cfg_exp['training']['iql_steps'])

    # 最终评估
    logger.info('开始评估...')
    metrics = trainer.evaluate(env, episodes=10)
    logger.info(f'评估指标: {metrics}')


if __name__ == '__main__':
    main()
