import numpy as np
import torch


def as_sparse_tensor(A):
    """Convert a sparse matrix to a sparse tensor.

    :param scipy.sparse.spmatrix A: Matrix to convert
    :rtype: torch.sparse.FloatTensor

    """
    sp = A.tocoo()
    return torch.sparse_coo_tensor(np.stack((sp.row, sp.col)), sp.data, sp.shape)
