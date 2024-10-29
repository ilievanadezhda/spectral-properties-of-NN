import torch


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)


def orthogonal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return w


def generate_random_vector(params):
    """
    generate a random vector with the same shape as the parameters
    """
    return [torch.randn_like(p) for p in params]


def generate_random_rademacher_vector(params):
    """
    generate a random Rademacher vector with the same shape as the parameters
    """
    return [torch.randint(0, 2, p.size(), device=p.device, dtype=p.dtype)*2-1 for p in params]


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

