import numpy as np
import numpy.testing
import scipy as sp
import scipy.sparse

from structure2vec.sparse_tensor import as_sparse_tensor


def test_as_sparse_tensor():
    spmatrix = sp.sparse.eye(4)
    sptensor = as_sparse_tensor(spmatrix)
    np.testing.assert_array_equal(spmatrix.todense(), sptensor.to_dense().numpy())
