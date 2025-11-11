#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
评估入口：加载模型并在固定场景下评估。
"""
import os
import yaml
from src.utils.logger import get_logger
from src.environment.wta_env import WTAEnv
from src.models.hdt_iql import HDTIQLPolicy
from src.algorithms.evaluator import run_eval


def load_yaml(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return yaml.safe_load(f)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(root, '..'))
    cfg_env = load_yaml(os.path.join(project_root, 'config', 'env.yaml'))
    cfg_reward = load_yaml(os.path.join(project_root, 'config', 'reward.yaml'))
    cfg_model = load_yaml(os.path.join(project_root, 'config', 'model.yaml'))

    log_dir = os.path.join(project_root, 'experiments', 'results', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger('eval', os.path.join(log_dir, 'eval.log'))

    env = WTAEnv(cfg_env, cfg_reward, cfg_model)
    policy = HDTIQLPolicy(cfg_model)
    # TODO: 可选加载权重

    metrics = run_eval(policy, env, episodes=10)
    logger.info(f'评估指标: {metrics}')


if __name__ == '__main__':
    main()