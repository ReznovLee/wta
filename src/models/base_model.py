import os
import torch


class BaseModel:
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)