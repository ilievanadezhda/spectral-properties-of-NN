import torch

def normalization(v):
    """
    normalize a vector
    """
    v = v / (torch.norm(v) + 1e-6)
    return v

def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = w - torch.dot(w, v) * v
    return normalization(w)