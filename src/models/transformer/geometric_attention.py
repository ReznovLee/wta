import torch


def apply_geometric_bias(attn_scores, features):
    return attn_scores