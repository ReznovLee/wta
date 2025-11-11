# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Generate offline datasets using ScenarioGenerator and save to data/trajectories.
"""
import os
import yaml
from src.utils.logger import get_logger
from src.algorithms.dataset_builder import build_offline_dataset


def load_yaml(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return yaml.safe_load(f)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(root, '..'))
    cfg_data = load_yaml(os.path.join(project_root, 'config', 'data.yaml'))
    logger = get_logger('generate_dataset', os.path.join(project_root, 'experiments/results/logs', 'generate_dataset.log'))
    logger.info('Building offline dataset...')
    build_offline_dataset(cfg_data['scenario'], cfg_data, cfg_data['offline_dataset']['root'])
    logger.info('Dataset generation complete.')


if __name__ == '__main__':
    main()