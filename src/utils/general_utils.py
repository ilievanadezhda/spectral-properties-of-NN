import torch
import numpy as np


def lp_distance(v1, v2, p=2):
    """
    Compute the L_p (norm) distance between two vectors (tensors).

    Args:
        v1 (torch.Tensor): first vector
        v2 (torch.Tensor): second vector
        p (int): p-norm
    
    Returns:
        torch.Tensor: L_p distance between the two vectors
    """
    v1 = torch.sort(v1).values
    v2 = torch.sort(v2).values
    return torch.norm(v1 - v2, p)


def count_correctly_estimated_eigenvalues(lanczos_eigenvalues, gt_eigenvalues, ITER, tolerance=1e-5):
    """
    Count the number of eigenvalues that are correctly estimated.

    Args:
        lanczos_eigenvalues (torch.Tensor): Lanczos eigenvalues
        gt_eigenvalues (torch.Tensor): groundtruth eigenvalues
        ITER (int): number of iterations
        tolerance (float): tolerance for comparison

    Returns:
        int: number of elements that are the same within the given tolerance    
    """
    # sort the groundtruth eigenvalues
    sorted_gt_eigenvalues = np.sort(gt_eigenvalues.numpy())
    # sort the Lanczos eigenvalues
    sorted_lanczos_eigenvalues = np.sort(lanczos_eigenvalues)
    # extract the first and last ITER // 2 eigenvalues from the groundtruth
    x = ITER // 2
    first_x_gt_eigenvalues = sorted_gt_eigenvalues[:x]
    last_x_gt_eigenvalues = sorted_gt_eigenvalues[-x:]
    # combine the first and last ITER // 2 eigenvalues
    combined_gt_eigenvalues = np.concatenate((first_x_gt_eigenvalues, last_x_gt_eigenvalues))
    # count the number of elements that are the same within the given tolerance
    same_count = np.sum(np.abs(sorted_lanczos_eigenvalues - combined_gt_eigenvalues) < tolerance)
    return same_count