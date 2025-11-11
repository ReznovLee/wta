import torch
import torch.nn as nn


class PointerDecoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

    def decode(self, query_set, key_set, action_masks, top_k=32):
        return None