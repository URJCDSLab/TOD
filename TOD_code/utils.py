"""
Common functions to evaluate the methods proposed. Implementation optimized
using numba.
"""
from itertools import combinations

import numpy as np
from numba import njit


def distance_matrix(x):
    # Compute the euclidean distance matrix.
    m, k = x.shape
    result = np.sum(np.abs(x[np.newaxis, :, :] - x[:, np.newaxis, :])**2,
                    axis=-1)**(1./2)
    return result


@njit()
def max_min_scale(mat):
    # Matrix min-max rescale.
    z_max = mat.max()
    z_min = mat.min()
    for i in range(mat.shape[0]):
        mat[i, :] = (mat[i, :] - z_min)/(z_max - z_min)
    return mat


@njit()
def nb_fill_diagonal(mat, val):
    # Numba optimization of diagonal fill.
    np.fill_diagonal(mat, val)


@njit()
def cross_product(distmat_k):
    # Computation of matrices using k-distances.
    n = distmat_k.shape[1]
    distmat = np.zeros((n, n), dtype=np.float32)
    distmat_k_T = distmat_k.T
    for i in range(n):
        distmat[i, :] = distmat_k * distmat_k_T[i, :]
    nb_fill_diagonal(distmat, 0)
    return distmat


@njit()
def triang_inequality(z_pos_triang, matrx):
    """Analysis of the triangle inequality.

    Parameters
    ----------
    z_pos_triang : ndarray
        Array of possible combinations of three elements.
    matrx : ndarray
        Matrix in which to analyze possible breaks.

    Returns
    -------
    cntMat : ndarray
        Matrix of breaks for each element in the matrix.
    """
    # Initialize matrix of breaks.
    cntMat = np.zeros(matrx.shape, dtype=np.int64)
    # Selection of positions of the matrix.
    ij = z_pos_triang[:, np.array([0, 1])]
    jk = z_pos_triang[:, np.array([1, 2])]
    ik = z_pos_triang[:, np.array([0, 2])]
    # Selection of distances in the matrix.
    d1 = np.array([matrx[i_ix, j_ix] for i_ix, j_ix in ij], dtype=np.float64)
    d2 = np.array([matrx[j_ix, k_ix] for j_ix, k_ix in jk], dtype=np.float64)
    d3 = np.array([matrx[i_ix, k_ix] for i_ix, k_ix in ik], dtype=np.float64)
    # Analysis of triangle inequality breaks.
    z_breaks = np.where((d1 + d2 < d3) | (d2 + d3 < d1) | (d1 + d3 < d2))[0]
    z_breaks = z_pos_triang[z_breaks]
    # Store of breaks in the matrix.
    for i_iter, j_iter, k_iter in z_breaks:
        cntMat[i_iter, j_iter] += 1
        cntMat[j_iter, k_iter] += 1
        cntMat[i_iter, k_iter] += 1
    cntMat = cntMat + cntMat.T
    return cntMat


@njit()
def nb_sum(x):
    # Numba optimization of sum of breaks.
    y = np.sum(x, 1)
    return y


@njit()
def random_choice(i, n_raw, z_size):
    # Numba optimization of random sampling.
    np.random.seed(1937*i)
    id_test = np.random.choice(np.array(list(range(n_raw)), dtype=np.int64), size=z_size, replace=False)
    id_test.sort()
    return id_test

