import torch


def lp_distance(v1, v2, p=2):
    """
    Compute the L_p (norm) distance between two vectors (tensors).
    """
    v1 = torch.sort(v1).values
    v2 = torch.sort(v2).values
    return torch.norm(v1 - v2, p)
