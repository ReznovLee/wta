import numpy as np
from typing import Dict, Optional, Tuple


class EKFTracker:
    """
    扩展卡尔曼滤波（EKF）预测器（简化版）：
    - 目标：为 physics_engine 提供统一的未来预测接口，用于估计目标在 tau 秒后的期望位置与速度。
    - 状态定义：x = [px, py, pz, vx, vy, vz]^T（常速模型）。
    - 过程模型（线性近似）：
        px' = px + vx * dt
        py' = py + vy * dt
        pz' = pz + vz * dt
        vx' = vx
        vy' = vy
        vz' = vz
      对应状态转移矩阵 F(dt)。
    - 噪声：使用对角高斯过程噪声 Q(dt)，默认为较小值；若 target_meta 提供类型或参数，则可适度放大相关方向的噪声（例如巡航的横向扰动、弹道的竖直方向）。

    接口：
      predict_horizon(current_pos: np.ndarray, current_vel: np.ndarray, tau: float, target_meta: Dict) -> Dict
        返回 {'position': np.ndarray(shape=(3,)), 'velocity': np.ndarray(shape=(3,))}

    说明：
    - 本实现以“预测”为主，不包含测量更新（update）；若后续需要接入传感器量测，可扩展 add_measurement / update 方法。
    - 该预测器是稳健的基线版本：它不会试图完全重建 physics_engine 内的非线性运动模型（如斜抛与横向正弦）；如需更贴近特定模型，可启用 model_aided 简单修正（见下）。
    """

    def __init__(self,
                 dt: float = 0.5,
                 q_pos_std: float = 1.0,
                 q_vel_std: float = 0.5,
                 use_model_aided: bool = True,
                 seed: Optional[int] = None):
        self.dt = float(dt)
        self.q_pos_std = float(q_pos_std)
        self.q_vel_std = float(q_vel_std)
        self.use_model_aided = bool(use_model_aided)
        self.rng = np.random.RandomState(seed if seed is not None else 42)
        # 初始协方差（可选用于不确定性传播；当前预测不强依赖协方差）
        self.P = np.eye(6) * 1.0

    # ------------------------------
    # 公有接口：未来预测
    # ------------------------------
    def predict_horizon(self,
                        current_pos: np.ndarray,
                        current_vel: np.ndarray,
                        tau: float,
                        target_meta: Optional[Dict] = None) -> Dict:
        pos0 = np.asarray(current_pos, dtype=float).reshape(3)
        vel0 = np.asarray(current_vel, dtype=float).reshape(3)
        tau = float(tau)
        if tau <= 0.0:
            return {'position': pos0.copy(), 'velocity': vel0.copy()}

        # 初始状态
        x = np.zeros(6, dtype=float)
        x[0:3] = pos0
        x[3:6] = vel0

        steps = max(1, int(np.ceil(tau / self.dt)))
        dt = tau / steps
        P = self.P.copy()

        for k in range(steps):
            # 基础常速状态转移
            F = self._state_transition(dt)
            x_prev = x.copy()
            x = F @ x
            # 简化增强：按目标类型施加过程模型的修正（不改变接口）
            if self.use_model_aided and target_meta is not None:
                x = self._apply_model_aided(x, dt, target_meta, step_index=k, x_prev=x_prev)
            # 协方差传播（可选）
            Q = self._process_noise(dt, target_meta)
            P = F @ P @ F.T + Q

        pos_pred = x[0:3].copy()
        vel_pred = x[3:6].copy()
        return {'position': pos_pred, 'velocity': vel_pred}

    # ------------------------------
    # 内部：模型辅助微修正
    # ------------------------------
    def _apply_model_aided(self, x: np.ndarray, dt: float, meta: Dict, step_index: int, x_prev: Optional[np.ndarray] = None) -> np.ndarray:
        """按目标类型施加简化增强的过程模型修正。
        注意：基础 CV 更新已通过 F@x 完成。此处仅添加相对 CV 的“额外修正”，避免重复积分。
        - ballistic：额外竖直位移校正 Δpz = -g*dt^2；速度校正 vz -= g*dt
        - cruise：平飞阶段添加横向位移；俯冲阶段竖直速度减少 dive_rate，并添加竖直位移校正 Δpz = -dive_rate*dt
        - aircraft：速度绕 z 轴旋转，位置添加因速度变化带来的位移校正 Δp = (v_rot - v_old) * dt
        """
        ttype = str(meta.get('type', 'cruise'))
        mspec = meta.get('motion_specs', {}) if isinstance(meta.get('motion_specs', {}), dict) else {}
        if x_prev is None:
            x_prev = x.copy()

        if ttype == 'ballistic':
            g = float(mspec.get('ballistic', {}).get('gravity_mps2', 9.81))
            # 位置额外校正：相对 CV 的垂直位移
            x[2] += (-g) * (dt ** 2)
            # 速度校正
            x[5] -= g * dt
            return x

        if ttype == 'cruise':
            cruise_params = mspec.get('cruise', {})
            amp = float(cruise_params.get('lateral_amp_m', 0.0))
            omega = float(cruise_params.get('lateral_omega', 0.0))
            dive_threshold = float(cruise_params.get('dive_threshold_m', 3000.0))
            dive_rate = float(cruise_params.get('dive_rate_mps', 0.0))
            dive_alt_th = float(cruise_params.get('dive_altitude_threshold_m', 1000.0))
            trigger_mode = str(cruise_params.get('dive_trigger_mode', 'distance_xyz'))
            impact = np.array(meta.get('impact_point', x_prev[0:3]), dtype=float)
            p = x_prev[0:3]
            dist_xyz = float(np.linalg.norm(impact - p))
            dist_xy = float(np.linalg.norm(impact[0:2] - p[0:2]))

            def should_cruise() -> bool:
                if trigger_mode == 'distance_xy':
                    return dist_xy > dive_threshold
                if trigger_mode == 'altitude':
                    return float(p[2]) > dive_alt_th
                return dist_xyz > dive_threshold

            if should_cruise():
                # 横向扰动位移（正交于速度的方向），不改速度模长
                v = x_prev[3:5]
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-9 and amp > 0.0 and omega > 0.0:
                    ortho = np.array([-v[1], v[0]], dtype=float)
                    ortho_norm = np.linalg.norm(ortho)
                    if ortho_norm > 1e-9:
                        ortho = ortho / ortho_norm
                        lateral_2d = amp * np.sin(omega * (step_index + 1) * dt) * ortho
                        x[0] += lateral_2d[0]
                        x[1] += lateral_2d[1]
                return x
            else:
                # 俯冲：竖直速度减少，位置额外校正
                if dive_rate > 0.0:
                    x[5] -= dive_rate
                    x[2] += (-dive_rate) * dt
                return x

        if ttype == 'aircraft':
            w = float(mspec.get('aircraft', {}).get('turn_rate_rad_s', 0.0))
            if abs(w) > 1e-9:
                vx_old, vy_old = x_prev[3], x_prev[4]
                cosw = np.cos(w * dt)
                sinw = np.sin(w * dt)
                vx_rot = vx_old * cosw - vy_old * sinw
                vy_rot = vx_old * sinw + vy_old * cosw
                # 速度更新
                x[3] = vx_rot
                x[4] = vy_rot
                # 位置额外校正：使用速度变化带来的位移增量
                x[0] += (vx_rot - vx_old) * dt
                x[1] += (vy_rot - vy_old) * dt
            return x

        # 其它类型或未知类型：不处理
        return x

    # ------------------------------
    # 内部：状态转移与噪声
    # ------------------------------
    def _state_transition(self, dt: float) -> np.ndarray:
        F = np.eye(6, dtype=float)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def _process_noise(self, dt: float, meta: Optional[Dict]) -> np.ndarray:
        # 基础对角噪声（随类型微调）
        q_pos = self.q_pos_std
        q_vel = self.q_vel_std
        if meta is not None:
            ttype = str(meta.get('type', 'cruise'))
            if ttype == 'cruise':
                q_pos *= 1.5
                q_vel *= 1.2
            elif ttype == 'ballistic':
                # 竖直方向更不确定
                pass
            elif ttype == 'aircraft':
                q_pos *= 1.2
        Q = np.zeros((6, 6), dtype=float)
        Q[0, 0] = (q_pos * dt) ** 2
        Q[1, 1] = (q_pos * dt) ** 2
        Q[2, 2] = (q_pos * dt) ** 2
        Q[3, 3] = (q_vel * dt) ** 2
        Q[4, 4] = (q_vel * dt) ** 2
        Q[5, 5] = (q_vel * dt) ** 2
        return Q


__all__ = ['EKFTracker']