import numpy as np


def build_action_masks(state, env_cfg):
    """
    构造拦截器-目标分配的硬约束掩码。
    返回: dict，包含形状为 [num_interceptors, num_targets] 的布尔矩阵。
    """
    num_interceptors = state.get('num_interceptors', 0)
    num_targets = state.get('num_targets', 0)
    assign_mask = np.ones((num_interceptors, num_targets), dtype=bool)
    range_mask = np.ones_like(assign_mask)
    ammo_mask = np.ones_like(assign_mask)
    time_window_mask = np.ones_like(assign_mask)

    # 简化约束示例：弹药为0的拦截器不可分配
    if 'interceptor_ammo' in state:
        ammo = np.array(state['interceptor_ammo'])
        for i in range(num_interceptors):
            if ammo[i] <= 0:
                ammo_mask[i, :] = False

    masks = {
        'assign_mask': assign_mask,
        'range_mask': range_mask,
        'ammo_mask': ammo_mask,
        'time_window_mask': time_window_mask,
    }
    return masks