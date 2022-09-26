import networkx as nx
import numpy as np
import numpy.testing
import pytest

from structure2vec.graph_collate import Graph, graph_collate, reduction_from_orders
from structure2vec.graphs import e2n_construct, n2n_construct


@pytest.mark.parametrize(
    "mode,reduction",
    [
        (
            "sum",
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
            ],
        ),
        (
            "mean",
            [
                [1 / 2, 1 / 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1 / 4, 1 / 4, 1 / 4, 1 / 4],
            ],
        ),
        (
            "first",
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
            ],
        ),
    ],
)
def test_reduction_from_orders(mode, reduction):
    actual = reduction_from_orders([2, 3, 4], mode=mode)
    np.testing.assert_array_equal(
        np.array(reduction, dtype=np.float32), actual.to_dense().numpy()
    )


def _dummy_mfe_observations(order, edges=True):
    graph = nx.path_graph(order)
    structure = []
    features = []
    structure.append(n2n_construct(graph))
    features.append(np.zeros((order, 5), dtype=np.float32))
    if edges:
        structure.append(e2n_construct(graph))
        features.append(np.zeros((order - 1, 6)))
    label = np.array([1, 0], dtype=np.float32)
    return (Graph(tuple(structure), tuple(features)), label)


def test_graph_collate_edges():
    batch = [_dummy_mfe_observations(3), _dummy_mfe_observations(4)]
    torch_batch = graph_collate(batch)
    torch_obs, torch_labels = torch_batch
    assert torch_labels.shape == (2, 2)
    (reduction, n2n_block, e2n_block), (node_features, edge_features) = torch_obs
    assert reduction.shape == (2, 7)
    assert n2n_block.shape == (7, 7)
    assert node_features.shape == (7, 5)
    assert e2n_block.shape == (7, 5)
    assert edge_features.shape == (5, 6)


def test_graph_collate_no_edges():
    batch = [
        _dummy_mfe_observations(3, edges=False),
        _dummy_mfe_observations(4, edges=False),
    ]
    torch_batch = graph_collate(batch)
    torch_obs, torch_labels = torch_batch
    assert torch_labels.shape == (2, 2)
    (reduction, n2n_block), (node_features,) = torch_obs
    assert reduction.shape == (2, 7)
    assert n2n_block.shape == (7, 7)
    assert node_features.shape == (7, 5)


def test_graph_collate_parallel_graphs():
    batch = [
        (_dummy_mfe_observations(2)[0], *_dummy_mfe_observations(3)),
        (_dummy_mfe_observations(4)[0], *_dummy_mfe_observations(5)),
    ]
    torch_batch = graph_collate(batch)
    graph1, graph2, labels = torch_batch
    assert isinstance(graph1, Graph)
    assert isinstance(graph2, Graph)
