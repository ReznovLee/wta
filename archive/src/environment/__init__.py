"""
Alias package for correctly spelled 'environment'.
This module re-exports key components from src.enviroment.

Note: We keep src.enviroment intact to avoid breaking existing imports.
Use this package for new code to ensure correct spelling.
"""

from src.environment.wta_env import WTAEnv
from src.environment.scenario_generator import ScenarioGenerator
from src.environment.physics_engine import PhysicsEngine
from src.environment.reward_calculator import RewardCalculator

__all__ = [
    "WTAEnv",
    "ScenarioGenerator",
    "PhysicsEngine",
    "RewardCalculator",
]