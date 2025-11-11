import os
from src.environment.scenario_generator import load_config, generate_random_targets


def build_offline_dataset(scenario_cfg, data_cfg, output_root):
    os.makedirs(output_root, exist_ok=True)
    # Placeholder: would generate and serialize trajectories according to cfg
    return