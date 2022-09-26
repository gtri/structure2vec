import networkx as nx
import numpy as np
import pytest
from hypothesis import given, note
from hypothesis_networkx import graph_builder

from . import orig_graphs as graphs

s2v_lib = pytest.importorskip(
    "s2v_lib", reason="Need original s2v_lib to compare our implementation to it."
)


class S2VGraph:
    def __init__(self, g):
        self.num_nodes = len(g.nodes)

        x, y = zip(*g.edges())
        self.num_edges = len(x)
        edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        edge_pairs[:, 0] = x
        edge_pairs[:, 1] = y
        self.edge_pairs = edge_pairs.flatten()


@given(graph_builder(graph_type=nx.Graph, min_nodes=2, min_edges=1))
def test_n2n_construct(G):
    note(str(nx.nx_pydot.to_pydot(G)))
    n2n, e2n, subg = s2v_lib.S2VLIB.PrepareMeanField([S2VGraph(G)])
    np.testing.assert_equal(
        n2n.to_dense().numpy(), graphs.n2n_construct(G).astype(np.float32).todense()
    )


@given(graph_builder(graph_type=nx.Graph, min_nodes=2, min_edges=1))
def test_e2n_construct(G):
    note(str(nx.nx_pydot.to_pydot(G)))
    n2n, e2n, subg = s2v_lib.S2VLIB.PrepareMeanField([S2VGraph(G)])
    np.testing.assert_equal(
        e2n.to_dense().numpy(), graphs.e2n_construct(G).astype(np.float32).todense()
    )


@given(graph_builder(graph_type=nx.Graph, min_nodes=2, min_edges=1))
def test_n2e_construct(G):
    note(str(nx.nx_pydot.to_pydot(G)))
    n2e, e2e, e2n, subg = s2v_lib.S2VLIB.PrepareLoopyBP([S2VGraph(G)])
    np.testing.assert_equal(
        n2e.to_dense().numpy(), graphs.n2e_construct(G).astype(np.float32).todense()
    )


@given(graph_builder(graph_type=nx.Graph, min_nodes=2, min_edges=1))
def test_e2e_construct(G):
    note(str(nx.nx_pydot.to_pydot(G)))
    n2e, e2e, e2n, subg = s2v_lib.S2VLIB.PrepareLoopyBP([S2VGraph(G)])
    np.testing.assert_equal(
        e2e.to_dense().numpy(), graphs.e2e_construct(G).astype(np.float32).todense()
    )
