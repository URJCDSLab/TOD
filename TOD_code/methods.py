"""
Implementation of the methods proposed on Triangle-based Outlier Detection.

https://doi.org/10.1016/j.patrec.2022.03.008
"""
from itertools import combinations

import numpy as np
from operator import matmul

from TOD_functions.utils import distance_matrix, max_min_scale, nb_fill_diagonal
from TOD_functions.utils import cross_product, triang_inequality, nb_sum, random_choice



def dis_matrices(data, k_l):
    """Calculation of dissimilarity matrices to feed TOD and sTOD.

    Parameters
    ----------
    data : DataFrame
        Original dataset.
    k_l : list
        L k values to compute the L dissimilarity matrices.

    Returns
    -------
    distmats: ndarray
        Array of the L dissimilarity matrices.
    """
    # Number of elements to analyze.
    n = data.shape[0]
    if max(k_l) > n:
        raise Exception("Reduce the maximum k.")
    # Initialize array of matrices.
    distmats = np.zeros((len(k_l), n, n), dtype=np.float64)
    # Compute euclidean distance among elements and rescale them.
    dist_mat = distance_matrix(data)
    dist_mat = max_min_scale(dist_mat)
    # Sort elements per row to select the k-neighbor.
    dist_mat = np.sort(dist_mat, 1)
    # Selection of k-distances to k-neighbors.
    distmat_k_l = dist_mat[:, k_l]
    distmat_k_l = distmat_k_l.transpose()
    distmat_k_l = distmat_k_l.reshape((len(k_l), 1, n))
    # Compute of dissimilarity matrices.
    for ix, distmat_k in enumerate(distmat_k_l):
        # Dissimilarity matrix compute.
        distmats[ix] = cross_product(distmat_k)
        # Matrix rescale.
        distmats[ix] = max_min_scale(distmats[ix])
    return distmats



def TOD(distmats, th):
    """Triangle-based Outlier Detection.

    Parameters
    ----------
    distmats : ndarray
        Array of L dissimilarity matrices.
    th : float
        Proportion of outliers to select.

    Returns
    -------
    pred_out: ndarray
        Array of the possible outliers.
    """
    # Number of elements to analyze.
    n = distmats[0].shape[0]
    # Possible combinations of 3 elements.
    z_pos_triang = np.array(list(combinations(range(n), 3)))
    # Maximum and minimum matrices.
    matrix_max = distmats.max(0); matrix_min = distmats.min(0)
    # Analysis of the triangle inequality.
    cntMat_max = triang_inequality(z_pos_triang, matrix_max)
    cntMat_min = triang_inequality(z_pos_triang, matrix_min)
    # Count of breaks per element in each matrix.
    rot_max = nb_sum(cntMat_max)
    rot_min = nb_sum(cntMat_min)
    # Selection of threshold for filtering each array of breaks.
    th_max = np.quantile(rot_max, th)
    th_min = np.quantile(rot_min, th)
    # Selection of possible outliers.
    preds_max = rot_max >= th_max
    preds_min = rot_min >= th_min
    pred_out = np.array([-1 if x else 1 for x in preds_max & preds_min])
    return pred_out


def sTOD(distmats, th, z_size=50, k_iter=100):
    """Sampling TOD.

    Parameters
    ----------
    distmats : ndarray
        Array of L dissimilarity matrices.
    th : float
        Threshold to select possible outliers.
    z_size : int
        Size of the random sampling.
    k_iter : int
        Constant to compute the number of random samplings.

    Returns
    -------
    pred_out: ndarray
        Array of the possible outliers.
    """
    # Number of elements to analyze and required random samplings.
    n = distmats[0].shape[0]
    tot_iter = int(k_iter*n/z_size)
    # Initialize array of breaks.
    rot_max = np.zeros(n); rot_min = np.zeros(n)
    # Possible combinations of 3 elements.
    z_pos_triang = np.array(list(combinations(range(z_size), 3)))
    # Random sampling.
    for i in range(tot_iter):
        # Elements selection.
        items_iter = random_choice(i, n, z_size)
        distmats_iter = distmats[:, items_iter,:][:, :, items_iter]
        # Maximum and minimum matrices.
        matrix_max = distmats_iter.max(0); matrix_min = distmats_iter.min(0)
        # Analysis of the triangle inequality.
        cntMat_max = triang_inequality(z_pos_triang, matrix_max)
        cntMat_min = triang_inequality(z_pos_triang, matrix_min)
        # Count of breaks per element in each matrix.
        rot_max_iter = nb_sum(cntMat_max)
        rot_min_iter = nb_sum(cntMat_min)
        # Join of breaks.
        rot_max[items_iter] += rot_max_iter
        rot_min[items_iter] += rot_min_iter
    # Selection of threshold for filtering each array of breaks.
    th_max = np.quantile(rot_max, th)
    th_min = np.quantile(rot_min, th)
    # Selection of possible outliers.
    preds_max = rot_max >= th_max
    preds_min = rot_min >= th_min
    pred_out = np.array([-1 if x else 1 for x in preds_max & preds_min])
    return pred_out

