import random


class ReplayBuffer:
    """简单的循环队列式 Replay Buffer，存储字典类型的过渡数据。

    过渡结构建议包含以下键：
    - obs: 当前观测（字典）
    - action: 动作（list[(i,j)]）
    - reward: 浮点值
    - next_obs: 下一时刻观测（字典）
    - done: 终止标志
    - masks: 当前时刻的动作掩码（字典）
    - info: 环境返回的信息字典
    """

    def __init__(self, max_size=100000, seq_len=256):
        self.max_size = int(max_size)
        self.seq_len = int(seq_len)
        self.storage = []
        self._next_idx = 0

    def __len__(self):
        return len(self.storage)

    def add(self, transition):
        if len(self.storage) < self.max_size:
            self.storage.append(transition)
        else:
            # 覆盖最旧的元素（循环队列）
            self.storage[self._next_idx] = transition
        self._next_idx = (self._next_idx + 1) % self.max_size

    def sample(self, batch_size=64):
        if len(self.storage) == 0:
            return []
        bs = min(batch_size, len(self.storage))
        return random.sample(self.storage, bs)

    def clear(self):
        self.storage.clear()
        self._next_idx = 0