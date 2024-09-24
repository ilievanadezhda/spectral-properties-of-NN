#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import math
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm


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


### My functions ###
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


def get_hessian_hvp(model):
    """ 
    compute hessian using hessian-vector products
    """
    params, grads = get_params_grad(model)
    num_params = sum(p.numel() for p in params)
    hessian = np.empty((num_params, num_params))
    for i in tqdm(range(num_params)):
        v = [torch.zeros(p.size()) for p in params]
        # flatten the vector and set the i-th element to 1
        flattened_v = torch.zeros(num_params)
        flattened_v[i] = 1
        # reshape the vector back to the original shape
        reshaped_v = []
        start = 0
        for t in v:
            numel = t.numel()  # get the number of elements in the current tensor
            reshaped_v.append(flattened_v[start:start + numel].view_as(t))  # reshape the chunk back
            start += numel
        # compute the hessian-vector product
        hvp = hessian_vector_product(grads, params, reshaped_v)
        hvp_flat = torch.cat([t.flatten() for t in hvp])
        hessian[i] = hvp_flat.cpu().detach().numpy()
    return hessian.T