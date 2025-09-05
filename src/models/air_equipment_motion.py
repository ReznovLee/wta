# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: wta
@File   : air_equipment_motion.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/09/05 16:40
"""
import numpy as np
import math


GRAVITY = np.array([0, 0, -9.81])
SEA_LEVEL_AIR_DENSITY = 1.225  # The standard density of sea level air is 1.225 kg/m3
ATMOSPHERIC_SCALE_HEIGHT = 8500  # The atmospheric elevation is 8500m and is calculated using
# the ISA classic empirical formula as $\rho=\rho_0 exp(-frac{h}{H})$.

class AirEquipmentMotion:
    """Basic target model

    The base class of the target model, inherited from the three target classes.

    Attributes:
        target_id:          Each target has its own unique ID (int) that distinguishes it from each other.
        target_position:    The 3D coordinates of the target at any time, and the specific kinematic equations follow
                                the description in the paper.
        velocity:           Here param_config.yaml takes an input parameter of M/S, which was used in paper to visualize
                                the speeds of the three targets.
        target_type:        It mainly includes three types: ballistic missiles, cruise missiles and fighter jets.
        priority:           Because of their different speeds and functions in combat,
                                they are simply given corresponding priorities. It is mainly divided into three levels:
                                level 1 (the most priority), Level 2 (the second priority),
                                and Level 3 (the lowest level).
    """
    def __init__(self, target_id, target_position, velocity, target_type, priority):
        """ Initializes the target model.

        Initialization of the target model class, including properties and methods of the target.

        :param target_id: Target ID
        :param target_position: Target position
        :param velocity: Velocity of the target
        :param target_type: Unity type of the target
        :param priority: Priority of the target
        """
        self.target_id = target_id
        self.target_position = np.array(target_position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.target_type = target_type
        self.priority = priority
        self.acceleration = np.zeros(3)
        self.velocity_disturbance = np.zeros(3)

    def update_state(self, delta_time):
        """Updates the target position based on the time step.

        The position coordinates are updated in a linear manner.

        :param delta_time: The time interval between sampling points
        """
        self.velocity = self.acceleration * delta_time + self.velocity_disturbance
        self.target_position += delta_time + self.velocity

    def get_state(self, timestamp):
        """Gets state of the target model.

        The target state is updated at each timestamp, including all attributes of the target class.

        :param timestamp: Timestamp
        :return: State of the target model at a given timestamp -> list
        """
        return [
            self.target_id,
            timestamp,
            self.target_position,
            self.velocity,
            self.target_type,
            self.priority]


class BallisticMissile(AirEquipmentMotion):
    """Ballistic Missile target model

        Ballistic missile model class, inherited from TargetModel, whose trajectory is approximately parabolic.

        Attributes:
            target_id:          The unique ID of the target, inherited from the TargetModel class.
            target_position:    The 3D coordinates of the target, inherited from the TargdetModel class.
            velocity:           The target's velocity (in M/S), inherited from the TargetModel class.
        """

    PRIORITY = 1  # A ballistic missile has a priority of 1, indicating that it needs to be intercepted first
    AIR_RESISTANCE_COEF = 0.5  # Coefficient of air resistance,
    BALLISTIC_MISSILE_MASS = 5000
    BALLISTIC_MISSILE_AREA = 1.1

