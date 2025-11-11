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
import copy

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
    """ Ballistic Missile target model

        Ballistic missile model class, inherited from AirEquipmentMotion, whose trajectory is approximately parabolic.

        Attributes:
            target_id:          The unique ID of the target, inherited from the AirEquipmentMotion class.
            target_position:    The 3D coordinates of the target, inherited from the AirEquipmentMotion class.
            velocity:           The target's velocity (in M/S), inherited from the AirEquipmentMotion class.
        """

    PRIORITY = 1  # A ballistic missile has a priority of 1, indicating that it needs to be intercepted first
    AIR_RESISTANCE_COEF = 0.5  # Coefficient of air resistance,
    BALLISTIC_MISSILE_MASS = 5000  # The quality of the missile
    BALLISTIC_MISSILE_AREA = 1.1  # Radar cross-section of the missile

    def __init__(self, target_id, target_position, velocity):
        """ Initializes the Ballistic Missile class.

        Initialization of the ballistic missile class, including properties and methods of the target.
        The cruise missile is divided into active phase, interruption phase and reentry phase. Since the missile range
        is generally long, only part of the trajectory of the reentry phase is considered in this project.

        :param target_id: Target ID
        :param target_position: Target position
        :param velocity: Target speed
        """
        super().__init__(target_id, target_position, velocity, "Ballistic_Missile", self.PRIORITY)

    def _calculate_air_resistance_acceleration(self):
        """Calculates the air resistance acceleration of the missile.

        The acceleration of air resistance is approximately calculated by the classical formula of air resistance,
        and the formula of air density is obtained by ISA empirical formula.

        The formula is: $$\rho = $$

        :return: Air resistance acceleration
        """
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > 0:
            rho = SEA_LEVEL_AIR_DENSITY * math.exp(-self.target_position[2] / ATMOSPHERIC_SCALE_HEIGHT)
            resistance_magnitude = (0.5 * self.AIR_RESISTANCE_COEF * rho * velocity_magnitude * velocity_magnitude
                                    * self.BALLISTIC_MISSILE_AREA / self.BALLISTIC_MISSILE_MASS)
            velocity_direction = -self.velocity / velocity_magnitude
            return resistance_magnitude * velocity_direction
        return np.zeros(3)

    def update_state(self, delta_time):
        """Updates the target position based on the time step.

        The position coordinates are updated in a linear manner.

        :param delta_time: Time step
        """
        air_resistance = self._calculate_air_resistance_acceleration()
        self.acceleration = GRAVITY + air_resistance

        self.velocity += self.acceleration * delta_time
        self.target_position += self.velocity * delta_time


class CruiseMissile(AirEquipmentMotion):
    """ Cruise Missile target model

        Initialization of the cruise missile target model class, including properties and methods of the target.
        The cruise missile is divided into climbing phase, cruising phase and diving phase. The focus of this project is on
        radar tracking, so only the cruise phase and diving phase of the cruise missile are involved.

        Attributes:
            target_id:          The unique ID of the target, inherited from the TargetModel class.
            target_position:    The 3D coordinates of the target, inherited from the TargetModel class.
            velocity:           The target velocity (in M/S), inherited from the TargetModel class.
            cruise_end_point:   The cruise end point of the cruise missile phase.
            dive_time:          The dive time of the cruise phase.
            cruise_time:        The cruise time of the cruise phase.
            rocket_acceleration:The rocket acceleration of the cruise phase.
    """
    PRIORITY = 2
    CRUISE_ALTITUDE = 8000
    TRANSITION_DISTANCE = 3000  # Horizontal distance of the subduction section
    DISTURBANCE_SCALE = 0.5  # Disturbance factor

    def __init__(self, target_id, target_position, velocity, cruise_end_point, dive_time, cruise_time,
                 rocket_acceleration):
        """ Initializes the target model.

        Initialization of the cruise missile target model class, including properties and methods of the target.

        :param target_id: Target ID
        :param target_position: Target position
        :param velocity: M/S is the speed in units
        :param cruise_end_point: Cruise end point
        :param dive_time: Dive time
        :param cruise_time: Cruise time
        :param rocket_acceleration: Rocket acceleration in dive phase
        """
        super().__init__(target_id, target_position, velocity, "cruise_missile", self.PRIORITY)
        self.current_phase = "cruise"
        self.cruise_end_point = np.array(cruise_end_point)
        self.dive_time = dive_time
        self.cruise_time = cruise_time
        self.rocket_acceleration = rocket_acceleration
        self.acceleration = np.zeros(3)

    def _apply_cruise_control(self):
        """ Apply cruise control to the cruise phase.

        Cruise missile needs to add disturbance in both cruise and reentry phase to meet the actual operational
        requirements.
        """
        """
        height_error = self.CRUISE_ALTITUDE - self.target_position[2]
        normalized_height_error = np.clip(height_error / 100, -1, 1)
        height_correction = np.array([0, 0, normalized_height_error * self.DISTURBANCE_SCALE])

        horizontal_disturbance = np.random.normal(0, self.DISTURBANCE_SCALE, 2)
        disturbance = np.array([horizontal_disturbance[0], horizontal_disturbance[1], 0])

        return height_correction + disturbance
        """
        height_error = self.CRUISE_ALTITUDE - self.target_position[2]  # Height error calculation

        # Normalized height error, reducing the denominator has increased the response to the error
        normalized_height_error = np.clip(height_error / 10, -1, 1)

        # Adding sinusoidal oscillations in the height direction
        time_based_oscillation = np.sin(self.cruise_time * 0.1) * 0.3

        # Composite altitude correction
        height_correction = np.array([0, 0, (normalized_height_error + time_based_oscillation) *
                                      self.DISTURBANCE_SCALE])

        # Adding random horizontal disturbances
        horizontal_disturbance = np.random.normal(0, self.DISTURBANCE_SCALE * 0.7, 2)
        disturbance = np.array([horizontal_disturbance[0], horizontal_disturbance[1], 0])
        return height_correction + disturbance

    def _apply_dive_control(self):
        """ Apply dive control to the cruise phase.

        Cruise missile needs to add disturbance in both cruise and reentry phase to meet the actual operational
        requirements.
        """
        acceleration = np.random.normal(0, self.DISTURBANCE_SCALE, 3)
        return GRAVITY + self.rocket_acceleration + acceleration

    def _check_phase_transition(self, current_position):
        """ Check if missile should transition from cruise to dive phase

        :param current_position: Current missile position
        :return: True if missile should transition, False otherwise
        """
        horizontal_distance = np.linalg.norm(current_position[:2] - self.cruise_end_point[:2])
        return (horizontal_distance <= self.TRANSITION_DISTANCE) or (self.cruise_time <= 0)

    def update_state(self, delta_time):
        """Updates the target position based on the time step.

        The position coordinates are updated in a linear manner.

        :param delta_time: Time step
        """
        self.cruise_time -= delta_time

        if self.current_phase == "cruise":
            cruise_control = self._apply_cruise_control()
            self.acceleration = cruise_control
            if self._check_phase_transition(self.target_position):
                self.current_phase = "dive"
                dive_control = self._apply_dive_control()
                self.acceleration = dive_control
        else:  # dive phase
            dive_control = self._apply_dive_control()
            self.acceleration = dive_control

        self.velocity += self.acceleration * delta_time
        self.target_position += self.velocity * delta_time


class AircraftTargetModel(AirEquipmentMotion):
    """Aircraft target model

    Initialization of the aircraft target model class, including properties and methods of the target.

    Attributes:
        target_id:          The unique ID of the aircraft target, inherited from the TargetModel class.
        target_position:    The 3D coordinates of the aircraft target, inherited from the TargetModel class.
        velocity:           The aircraft target velocity (in M/S), inherited from the TargetModel class.

    """
    PRIORITY = 3
    MIN_ALTITUDE = 5000
    MAX_ALTITUDE = 10000
    AIR_RESISTANCE_COEF = 0.1
    TURN_RATE_MAX = 0.1
    VERTICAL_ACCELERATION = 5
    SPEED_CONTROL_FACTOR = 0.5  # 增大速度控制因子
    MAX_ACCELERATION = 10  # 添加最大加速度限制
    DIRECTION_STABILITY = 0.95  # 方向稳定性因子
    MANEUVER_INTERVAL = 10.0  # 大幅度机动的时间间隔

    def __init__(self, target_id, target_position, velocity_ms):
        """ Initializes the aircraft target model.

        Initialization of the aircraft target model class, including properties and methods of the target.

        :param target_id: The unique ID of the aircraft target.
        :param target_position: The 3D coordinates of the aircraft target.
        :param velocity_ms: M/S is the speed in units.
        """
        super().__init__(target_id, target_position, velocity_ms, "Aircraft", self.PRIORITY)
        self.min_altitude = self.MIN_ALTITUDE
        self.max_altitude = self.MAX_ALTITUDE
        self.acceleration = np.zeros(3)
        self.yaw = np.random.uniform(0, 2 * np.pi)
        self.pitch = np.random.uniform(-np.pi / 6, np.pi / 6)
        self.target_speed = np.linalg.norm(velocity_ms)
        self.target_direction = self._calculate_direction_from_velocity(velocity_ms)
        self.time_since_last_maneuver = 0.0

    def _apply_speed_control(self):
        """Apply speed control to maintain target speed"""
        current_speed = np.linalg.norm(self.velocity)
        if current_speed > 0:
            speed_diff = self.target_speed - current_speed
            acceleration = self.velocity * (speed_diff * self.SPEED_CONTROL_FACTOR / current_speed)
            # 限制加速度大小
            acc_magnitude = np.linalg.norm(acceleration)
            if acc_magnitude > self.MAX_ACCELERATION:
                acceleration = acceleration * (self.MAX_ACCELERATION / acc_magnitude)
            return acceleration
        return np.zeros(3)

    def _calculate_air_resistance(self):
        """ Calculate air resistance.

        The air resistance is calculated based on the current velocity of the target.

        :return: Air resistance
        """
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > 0:
            resistance = -self.AIR_RESISTANCE_COEF * velocity_magnitude * self.velocity
            return resistance
        return np.zeros(3)

    def _calculate_direction_from_velocity(self, velocity):
        """计算速度向量对应的方向"""
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > 0:
            return velocity / velocity_magnitude
        return np.array([1.0, 0.0, 0.0])  # 默认方向

    def _apply_altitude_control(self):
        """ Apply altitude control to the aircraft target.

        :return: Altitude control acceleration
        """
        height_margin = 200
        height = self.target_position[2]

        if height < self.MIN_ALTITUDE + height_margin:
            return np.array([0, 0, self.VERTICAL_ACCELERATION])
        elif height > self.MAX_ALTITUDE - height_margin:
            return np.array([0, 0, -self.VERTICAL_ACCELERATION])
        return np.zeros(3)

    def _apply_maneuver(self, time_step):
        """ Apply maneuver to the aircraft target.

        :param time_step: Time step
        :return: Maneuver acceleration
        """
        # 更新机动计时器
        self.time_since_last_maneuver += time_step

        # 决定是否进行大幅度机动
        if self.time_since_last_maneuver >= self.MANEUVER_INTERVAL:
            # 大幅度机动
            yaw_change = np.random.uniform(-self.TURN_RATE_MAX, self.TURN_RATE_MAX)
            pitch_change = np.random.uniform(-self.TURN_RATE_MAX / 2, self.TURN_RATE_MAX / 2)
            self.time_since_last_maneuver = 0.0  # 重置计时器
        else:
            # 小幅度调整
            yaw_change = np.random.uniform(-self.TURN_RATE_MAX / 5, self.TURN_RATE_MAX / 5)
            pitch_change = np.random.uniform(-self.TURN_RATE_MAX / 10, self.TURN_RATE_MAX / 10)

        # 更新航向和俯仰角
        self.yaw += yaw_change
        self.pitch += np.clip(self.pitch + pitch_change, -np.pi / 4, np.pi / 4)  # 限制俯仰角范围

        # 计算新方向
        new_direction = np.array([
            np.cos(self.pitch) * np.cos(self.yaw),
            np.cos(self.pitch) * np.sin(self.yaw),
            np.sin(self.pitch)
        ])

        # 平滑过渡到新方向，保持一定的稳定性
        self.target_direction = self.DIRECTION_STABILITY * self.target_direction + \
                                (1 - self.DIRECTION_STABILITY) * new_direction
        self.target_direction = self.target_direction / np.linalg.norm(self.target_direction)

        # 减小扰动幅度，使轨迹更平滑
        disturbance = np.array([
            np.random.normal(0, 0.1),  # 减小x方向扰动
            np.random.normal(0, 0.1),  # 减小y方向扰动
            np.random.normal(0, 0.05)  # 减小z方向扰动
        ]) * time_step

        return self.target_direction, disturbance

    def update_state(self, time_step):
        """Updates the target position based on the time step.

        The position coordinates are updated in a linear manner.

        :param time_step: Time step
        """
        air_resistance = self._calculate_air_resistance()
        direction, disturbance = self._apply_maneuver(time_step)
        altitude_control = self._apply_altitude_control()
        speed_control = self._apply_speed_control()

        # 计算期望速度
        desired_velocity = direction * self.target_speed

        # 计算速度差异，使用更平滑的过渡
        velocity_diff = desired_velocity - self.velocity
        direction_correction = velocity_diff * 0.2  # 平滑系数

        # 合成加速度
        total_acceleration = (
                air_resistance +
                altitude_control +
                direction_correction / time_step +  # 方向修正
                disturbance / time_step +
                speed_control
        )

        # 限制加速度大小
        acc_magnitude = np.linalg.norm(total_acceleration)
        if acc_magnitude > self.MAX_ACCELERATION:
            total_acceleration = total_acceleration * (self.MAX_ACCELERATION / acc_magnitude)

        # 防止NaN和无穷大
        self.acceleration = np.nan_to_num(total_acceleration, nan=0.0, posinf=0.0, neginf=0.0)

        # 更新速度和位置
        self.velocity += self.acceleration * time_step
        self.target_position += self.velocity * time_step


class AntiWeapon:
    """Anti-Weapon model for ground-based missile defense system

    This class models ground-based anti-missile weapons that intercept airborne targets.
    It calculates interception points, launch angles, and timing parameters based on
    target motion prediction and the anti-weapon's capabilities.

    Attributes:
        weapon_id:          Unique identifier for the anti-weapon system
        weapon_position:    Ground position of the anti-weapon (z-coordinate is always 0)
        weapon_type:        Type of the anti-weapon, determining its characteristics
        avg_speed:          Average speed of the anti-missile (m/s)
        max_range:          Maximum effective range of the anti-weapon system (m)
        min_range:          Minimum effective range of the anti-weapon system (m)
        max_altitude:       Maximum altitude the anti-missile can reach (m)
    """

    # Define weapon type characteristics as a class attribute
    WEAPON_SPECIFICATIONS = {
        "AntiMissile1": {  # 原 interceptor_for_ballistic
            "avg_speed": 2500,  # 高速拦截弹，适合拦截弹道导弹
            "max_range": 200000,
            "min_range": 20000,
            "max_altitude": 100000
        },
        "AntiMissile2": {  # 原 interceptor_for_cruise
            "avg_speed": 1000,  # 中速拦截弹，适合拦截巡航导弹
            "max_range": 100000,
            "min_range": 5000,
            "max_altitude": 20000
        },
        "AntiMissile3": {  # 原 interceptor_for_aircraft
            "avg_speed": 800,   # 低速拦截弹，适合拦截飞机
            "max_range": 80000,
            "min_range": 2000,
            "max_altitude": 25000
        }
        # 可以在此处添加新的武器类型
    }

    def __init__(self, weapon_id, weapon_position, weapon_type):
        """Initializes the AntiWeapon model.

        :param weapon_id: Unique identifier for the anti-weapon
        :param weapon_position: Ground position [x, y, z] where z is always 0
        :param weapon_type: Type of the anti-weapon (e.g., "AntiMissile1", "AntiMissile2", or "AntiMissile3")
        """
        self.weapon_id = weapon_id
        # Ensure the weapon is on the ground (z=0)
        weapon_position_array = np.array(weapon_position, dtype=np.float64)
        weapon_position_array[2] = 0.0  # Force z-coordinate to be 0
        self.weapon_position = weapon_position_array
        self.weapon_type = weapon_type

        # Set weapon characteristics based on type
        specs = self.WEAPON_SPECIFICATIONS.get(weapon_type)
        if not specs:
            raise ValueError(f"未知的武器类型: {weapon_type}，可用类型: {list(self.WEAPON_SPECIFICATIONS.keys())}")

        self.avg_speed = specs["avg_speed"]
        self.max_range = specs["max_range"]
        self.min_range = specs["min_range"]
        self.max_altitude = specs["max_altitude"]

    def predict_target_position(self, target, time_to_future):
        """Predicts the future position of a target based on its current state using Extended Kalman Filter.
        
        This method uses appropriate EKF model based on target type for more accurate prediction.
        
        :param target: Target object (BallisticMissile, CruiseMissile, or AircraftTargetModel)
        :param time_to_future: Time in seconds to predict into the future
        :return: Predicted position of the target
        """
        from src.utils.filter import BallisticMissileEKF, CruiseMissileEKF, AircraftIMMEKF
        
        # 确定目标类型并选择合适的EKF模型
        if target.target_type == "Ballistic_Missile":
            ekf = BallisticMissileEKF(dt=0.1)  # 使用较小的时间步长
            # 设置初始状态向量 [x, y, z, vx, vy, vz, ax, ay, az]
            ekf.x = np.concatenate([
                target.target_position,
                target.velocity,
                target.acceleration
            ])
        elif target.target_type == "Cruise_Missile":
            ekf = CruiseMissileEKF(dt=0.1)
            # 设置初始状态向量 [x, y, z, vx, vy, vz, ax, ay, az]
            ekf.x = np.concatenate([
                target.target_position,
                target.velocity,
                target.acceleration
            ])
        else:  # 默认为飞机目标
            ekf = AircraftIMMEKF(dt=0.1)
            # 对于IMM-EKF，我们需要更新所有模型的状态
            # 设置CV模型的状态 [x, y, z, vx, vy, vz]
            ekf.filters[ekf.filters.keys()[0]].x = np.concatenate([
                target.target_position,
                target.velocity
            ])
            # 如果有CA模型，设置其状态 [x, y, z, vx, vy, vz, ax, ay, az]
            if len(ekf.filters) > 1:
                ekf.filters[ekf.filters.keys()[2]].x = np.concatenate([
                    target.target_position,
                    target.velocity,
                    target.acceleration
                ])
        
        # 设置初始协方差矩阵为适度不确定性
        if hasattr(ekf, 'P'):
            ekf.P = np.eye(ekf.state_dim) * 10.0
        
        # 预测未来位置
        steps = int(time_to_future / ekf.dt)
        for _ in range(steps):
            ekf.predict()
        
        # 返回预测的位置
        if hasattr(ekf, 'x'):
            return ekf.x[:3]  # 返回位置部分 [x, y, z]
        else:  # 对于IMM-EKF，返回组合估计
            combined_state = ekf._combine_estimates()
            return combined_state[:3]  # 返回位置部分 [x, y, z]
    
    def calculate_interception_time(self, target, current_time, max_prediction_time=300, time_step=0.1):
        """Calculates the interception time and point for a given target.
        
        Uses an iterative approach to find when the anti-missile can intercept the target.
        
        :param target: Target object to intercept
        :param current_time: Current simulation time
        :param max_prediction_time: Maximum time to look ahead for interception
        :param time_step: Time step for the iterative calculation
        :return: Dictionary with interception parameters or None if interception is impossible
        """
        for t in np.arange(0, max_prediction_time, time_step):
            # Predict target position at time t
            predicted_target_pos = self.predict_target_position(target, t)
            
            # Calculate distance from weapon to predicted target position
            distance = np.linalg.norm(predicted_target_pos - self.weapon_position)
            
            # Check if target is within range constraints
            if distance > self.max_range or distance < self.min_range:
                continue
                
            # Check if target altitude is within capability
            if predicted_target_pos[2] > self.max_altitude:
                continue
            
            # Calculate time needed for anti-missile to reach this point
            time_to_reach = distance / self.avg_speed
            
            # If the time to reach is approximately equal to the prediction time,
            # we've found an interception point
            if abs(time_to_reach - t) < time_step:
                # Calculate launch angle (elevation angle from horizontal)
                horizontal_distance = np.linalg.norm(predicted_target_pos[:2] - self.weapon_position[:2])
                elevation_angle = np.arctan2(predicted_target_pos[2], horizontal_distance)
                elevation_angle_degrees = np.degrees(elevation_angle)
                
                # Calculate azimuth angle (not required as per specifications, but included for completeness)
                azimuth = np.arctan2(predicted_target_pos[1] - self.weapon_position[1], 
                                     predicted_target_pos[0] - self.weapon_position[0])
                azimuth_degrees = np.degrees(azimuth) % 360
                
                return {
                    'interception_possible': True,
                    'target_id': target.target_id,
                    'interception_point': predicted_target_pos,
                    'time_to_interception': time_to_reach,  # Time from launch to interception
                    'launch_time': current_time,  # Current time is when we launch
                    'interception_time': current_time + time_to_reach,  # When interception occurs
                    'elevation_angle': elevation_angle_degrees,  # In degrees (0-90)
                    'azimuth_angle': azimuth_degrees,  # In degrees (0-360)
                    'distance': distance  # Distance from launcher to interception point
                }
        
        # If we get here, no interception point was found
        return {
            'interception_possible': False,
            'target_id': target.target_id,
            'reason': 'No viable interception solution found within constraints'
        }
    
    def can_intercept(self, target):
        """Checks if the target can be intercepted with current weapon position and capabilities.
        
        Different weapon types have different interception success rates based on target type and distance.
        
        :param target: Target object to check
        :return: Dictionary with interception probability and constraints information
        """
        # 定义不同武器类型对不同目标的基础拦截成功率
        BASE_INTERCEPTION_RATES = {
            "AntiMissile1": {  # 原 interceptor_for_ballistic
                "Ballistic_Missile": 0.85,  # 对弹道导弹的拦截成功率高
                "cruise_missile": 0.60,     # 对巡航导弹的拦截成功率中等
                "Aircraft": 0.70            # 对飞机的拦截成功率中高
            },
            "AntiMissile2": {  # 原 interceptor_for_cruise
                "Ballistic_Missile": 0.40,  # 对弹道导弹的拦截成功率低
                "cruise_missile": 0.80,     # 对巡航导弹的拦截成功率高
                "Aircraft": 0.65            # 对飞机的拦截成功率中等
            },
            "AntiMissile3": {  # 原 interceptor_for_aircraft
                "Ballistic_Missile": 0.30,  # 对弹道导弹的拦截成功率低
                "cruise_missile": 0.55,     # 对巡航导弹的拦截成功率中等
                "Aircraft": 0.75            # 对飞机的拦截成功率高
            }
        }
        
        # 计算距离
        distance = np.linalg.norm(target.target_position - self.weapon_position)
        
        # 检查基本约束条件
        range_ok = self.min_range <= distance <= self.max_range
        altitude_ok = target.target_position[2] <= self.max_altitude
        
        # 如果不满足基本约束条件，直接返回不可拦截
        if not (range_ok and altitude_ok):
            return {
                "can_intercept": False,
                "reason": "Target outside weapon constraints",
                "range_ok": range_ok,
                "altitude_ok": altitude_ok,
                "probability": 0.0
            }
        
        # 获取基础拦截成功率
        base_rate = BASE_INTERCEPTION_RATES.get(self.weapon_type, {}).get(target.target_type, 0.5)
        
        # 根据距离调整拦截成功率
        # 在最佳拦截范围内（30%-70%的最大范围）拦截成功率最高
        optimal_min = self.min_range + (self.max_range - self.min_range) * 0.3
        optimal_max = self.min_range + (self.max_range - self.min_range) * 0.7
        
        distance_factor = 1.0  # 默认距离因子
        
        if distance < optimal_min:
            # 距离太近，成功率降低
            distance_factor = 0.7 + 0.3 * (distance - self.min_range) / (optimal_min - self.min_range)
        elif distance > optimal_max:
            # 距离太远，成功率降低
            distance_factor = 0.7 + 0.3 * (self.max_range - distance) / (self.max_range - optimal_max)
        
        # 计算最终拦截成功率
        interception_probability = base_rate * distance_factor
        
        return {
            "can_intercept": True,
            "range_ok": range_ok,
            "altitude_ok": altitude_ok,
            "probability": interception_probability,
            "distance": distance,
            "weapon_type": self.weapon_type,
            "target_type": target.target_type
        }
    
    def calculate_optimal_launch_parameters(self, target, current_time, time_horizon=60, time_step=1.0):
        """Calculates optimal launch parameters for intercepting a target.
        
        This method looks ahead over a time horizon to find the best interception opportunity.
        
        :param target: Target object to intercept
        :param current_time: Current simulation time
        :param time_horizon: How far ahead in time to look for interception opportunities
        :param time_step: Time step for checking interception opportunities
        :return: Dictionary with optimal launch parameters or None if interception is impossible
        """
        best_solution = None
        best_score = float('inf')  # Lower is better
        
        # Check multiple future launch times
        for launch_delay in np.arange(0, time_horizon, time_step):
            # Calculate interception at this potential launch time
            future_time = current_time + launch_delay
            
            # Create a copy of the target to simulate its future state
            # This is a simplified approach - in a real system, you would use a proper prediction model
            target_copy = copy.deepcopy(target)
            
            # Update target state to the potential launch time
            for _ in np.arange(0, launch_delay, time_step):
                target_copy.update_state(time_step)
            
            # Calculate interception for this future launch time
            interception = self.calculate_interception_time(target_copy, future_time)
            
            if interception['interception_possible']:
                # Score this solution (lower is better)
                # Here we prioritize earlier interception and shorter flight times
                score = interception['interception_time'] + 0.5 * interception['time_to_interception']
                
                if score < best_score:
                    best_score = score
                    best_solution = interception
                    best_solution['launch_time'] = future_time  # Update with the actual launch time
        
        return best_solution
    
    def simulate_interception(self, target, launch_time, time_step=0.1, max_simulation_time=300):
        """Simulates the interception process with more detailed physics.
        
        This method provides a more accurate simulation of the interception by considering
        changing target position during the anti-missile flight.
        
        :param target: Target object to intercept
        :param launch_time: Time when the anti-missile is launched
        :param time_step: Simulation time step
        :param max_simulation_time: Maximum simulation time
        :return: Dictionary with detailed interception results
        """
        # Initial calculation to get approximate interception point
        initial_calc = self.calculate_interception_time(target, launch_time)
        
        if not initial_calc['interception_possible']:
            return initial_calc
        
        # Extract initial parameters
        initial_interception_point = initial_calc['interception_point']
        initial_time_to_interception = initial_calc['time_to_interception']
        
        # Calculate initial direction vector for the anti-missile
        direction = initial_interception_point - self.weapon_position
        direction = direction / np.linalg.norm(direction)
        
        # Simulate the interception process
        current_time = launch_time
        anti_missile_position = self.weapon_position.copy()
        target_copy = copy.deepcopy(target)
        
        for _ in range(int(max_simulation_time / time_step)):
            # Update target position
            target_copy.update_state(time_step)
            
            # Update anti-missile position (simplified - constant speed along initial direction)
            anti_missile_position += direction * self.avg_speed * time_step
            
            # Check if interception occurred
            distance_to_target = np.linalg.norm(anti_missile_position - target_copy.target_position)
            if distance_to_target < 10:  # Assuming 10m is close enough for interception
                return {
                    'interception_possible': True,
                    'target_id': target.target_id,
                    'interception_point': target_copy.target_position,
                    'time_to_interception': current_time - launch_time,
                    'launch_time': launch_time,
                    'interception_time': current_time,
                    'distance': np.linalg.norm(target_copy.target_position - self.weapon_position),
                    'simulation_detail': 'Detailed simulation with target movement during interception'
                }
            
            current_time += time_step
            
            # If we've gone significantly past the expected interception time, abort
            if current_time > launch_time + 1.5 * initial_time_to_interception:
                break
        
        return {
            'interception_possible': False,
            'target_id': target.target_id,
            'reason': 'Detailed simulation failed to achieve interception'
        }


