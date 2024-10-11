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


def generate_eigenvalues(size, dict_eigenvalues):
    tensor = torch.zeros(size)
    pos_extreme_indices = torch.randperm(size)[:dict_eigenvalues["num_pos_extremes"]]
    neg_extreme_indices = torch.randperm(size)[:dict_eigenvalues["num_neg_extremes"]]
    around_zero_indices = torch.randperm(size)[:dict_eigenvalues["num_around_zero"]] 
    pos_extreme_values = torch.empty(dict_eigenvalues["num_pos_extremes"]).uniform_(dict_eigenvalues["pos_low"], dict_eigenvalues["pos_high"])
    neg_extreme_values = torch.empty(dict_eigenvalues["num_neg_extremes"]).uniform_(dict_eigenvalues["neg_low"], dict_eigenvalues["neg_high"])
    around_zero_values = torch.empty(dict_eigenvalues["num_around_zero"]).uniform_(dict_eigenvalues["around_zero_low"], dict_eigenvalues["around_zero_high"])
    tensor[pos_extreme_indices] = pos_extreme_values
    tensor[neg_extreme_indices] = neg_extreme_values
    tensor[around_zero_indices] = around_zero_values
    return tensor

def lp_distance(v1, v2, p=2):
    """ 
    compute the L_p (norm) distance between two vectors (tensors).
    warning: the vectors are sorted before computing the distance.
    """
    v1 = torch.sort(v1).values
    v2 = torch.sort(v2).values
    return torch.norm(v1 - v2, p)