class ReplayBuffer:
    def __init__(self, max_size=100000, seq_len=256):
        self.max_size = max_size
        self.seq_len = seq_len
        self.storage = []

    def add(self, transition):
        if len(self.storage) >= self.max_size:
            self.storage.pop(0)
        self.storage.append(transition)

    def sample(self, batch_size=64):
        return self.storage[:batch_size]