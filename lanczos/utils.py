import torch


def normalization(v):
    """
    normalize a vector
    """
    norm = torch.sqrt(torch.dot(v, v))
    norm = norm.cpu().item()
    v = v / (norm + 1e-6)
    return v


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = w - torch.dot(w, v) * v
    return normalization(w)


def orthogonal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    """
    for v in v_list:
        w = w - torch.dot(w, v) * v
    return w


def form_tridiagonal_mtx(alpha_list, beta_list, device):
    """
    form a tridiagonal matrix with alpha_list and beta_list
    """
    k = len(alpha_list)
    T = torch.zeros(k, k).to(device)
    for i in range(len(alpha_list)):
        T[i, i] = alpha_list[i]
        if i < len(alpha_list) - 1:
            T[i + 1, i] = beta_list[i]
            T[i, i + 1] = beta_list[i]
    return T


def generate_eigenvalues(size, dict_eigenvalues):
    """
    generate a tensor of eigenvalues with specified number of positive and negative extremes and values around zero
    """
    tensor = torch.zeros(size)
    # choose random indices
    indices = torch.randperm(size)[:dict_eigenvalues["num_around_zero"] + dict_eigenvalues["num_pos_extremes"] + dict_eigenvalues["num_neg_extremes"]]
    # take the first num_pos_extremes indices and assign them a random value between pos_low and pos_high
    pos_extreme_indices = indices[:dict_eigenvalues["num_pos_extremes"]]
    pos_extreme_values = torch.empty(dict_eigenvalues["num_pos_extremes"]).uniform_(dict_eigenvalues["pos_low"], dict_eigenvalues["pos_high"])
    # take the next num_neg_extremes indices and assign them a random value between neg_low and neg_high
    neg_extreme_indices = indices[dict_eigenvalues["num_pos_extremes"]:dict_eigenvalues["num_pos_extremes"] + dict_eigenvalues["num_neg_extremes"]]
    neg_extreme_values = torch.empty(dict_eigenvalues["num_neg_extremes"]).uniform_(dict_eigenvalues["neg_low"], dict_eigenvalues["neg_high"])
    # take the last num_around_zero indices and assign them a random value between around_zero_low and around_zero_high
    around_zero_indices = indices[dict_eigenvalues["num_pos_extremes"] + dict_eigenvalues["num_neg_extremes"]:]
    around_zero_values = torch.empty(dict_eigenvalues["num_around_zero"]).uniform_(dict_eigenvalues["around_zero_low"], dict_eigenvalues["around_zero_high"])
    # assign the values to the tensor
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