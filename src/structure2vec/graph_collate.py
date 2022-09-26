"""Graph data collation function."""

import collections
import collections.abc
import re

import numpy
import scipy.sparse
import torch

from structure2vec.sparse_tensor import as_sparse_tensor

Graph = collections.namedtuple("Graph", ["structure", "features"])


def _reduction_sum(orders):
    """Reduction implementation for sum.

    For example::
        >>> reduction_from_orders([2, 3, 4], mode='sum').to_dense().numpy()
        np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 1]])

    """
    I = []
    for i, order in enumerate(orders):
        I.extend([i] * order)
    J = list(range(len(I)))
    indexes = numpy.array([I, J])
    data = numpy.ones(len(I), dtype=numpy.float32)
    return indexes, data


def _reduction_mean(orders):
    """Reduction implementation for mean.

    For example::
        >>> reduction_from_orders([2, 3, 4], mode='mean').to_dense().numpy().T
        array([[0.5       , 0.        , 0.        ],
               [0.5       , 0.        , 0.        ],
               [0.        , 0.33333334, 0.        ],
               [0.        , 0.33333334, 0.        ],
               [0.        , 0.33333334, 0.        ],
               [0.        , 0.        , 0.25      ],
               [0.        , 0.        , 0.25      ],
               [0.        , 0.        , 0.25      ],
               [0.        , 0.        , 0.25      ]])
    """
    I = []
    D = []
    for i, order in enumerate(orders):
        I.extend([i] * order)
        D.extend([1 / order] * order)
    J = list(range(len(I)))
    indexes = numpy.array([I, J])
    data = numpy.array(D, dtype=numpy.float32)
    return indexes, data


def _reduction_first(orders):
    """Reduction implementation for first.

    For example::
        >>> reduction_from_orders([2, 3, 4], mode='first').to_dense().numpy()
        np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0]])

    """
    I = list(range(len(orders)))
    j = 0
    J = []
    for order in orders:
        J.append(j)
        j += order
    indexes = numpy.array([I, J])
    data = numpy.ones(len(I), dtype=numpy.float32)
    return indexes, data


REDUCTIONS = {
    "sum": _reduction_sum,
    "mean": _reduction_mean,
    "first": _reduction_first,
}


def reduction_from_orders(orders, *, mode="sum"):
    """Compute the reduction matrix as a tensor.

    The reduction matrix is used as the last step of the embedding
    forward pass to combine the embeddings for all of the nodes in a
    graph into a single vector for each graph.

    """
    batch_size = len(orders)
    total_nodes = sum(orders)
    indexes, data = REDUCTIONS[mode](orders)
    return torch.sparse_coo_tensor(indexes, data, (batch_size, total_nodes))


def block_diag_collate(batch, *, mode="sum"):
    columns = list(zip(*batch))
    result = [reduction_from_orders([x.shape[0] for x in columns[0]], mode=mode)]
    for column in columns:
        result.append(as_sparse_tensor(scipy.sparse.block_diag(column)))
    return result


def concatenate_collate(batch):
    columns = zip(*batch)
    return [torch.as_tensor(numpy.concatenate(col, axis=0)) for col in columns]


def graph_marker_collate(batch, mode="sum"):
    structure, features = zip(*batch)
    return Graph(
        block_diag_collate(structure, mode=mode), concatenate_collate(features)
    )


# The code below is based on torch.utils.data._utils.collate.default_collate.
# PyTorch is licensed with the 3 Clause BSD license.
# See LICENSE.pytorch at the root of this repository.

np_str_obj_array_pattern = re.compile(r"[SaUO]")

graph_collate_err_msg_format = (
    "graph_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def graph_collate(batch, *, mode="sum"):  # noqa: C901 # pragma: no cover
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(graph_collate_err_msg_format.format(elem.dtype))

            return graph_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, Graph):
        return graph_marker_collate(batch, mode=mode)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: graph_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(graph_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [graph_collate(samples) for samples in transposed]

    raise TypeError(graph_collate_err_msg_format.format(elem_type))
