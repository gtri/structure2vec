import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
from hypothesis import given, note
from hypothesis_networkx import graph_builder

from structure2vec import graphs

from . import orig_graphs


@given(graph_builder(graph_type=nx.Graph, min_nodes=2, min_edges=1))
def test_n2n_construct(G):
    note(str(nx.nx_pydot.to_pydot(G)))
    old = orig_graphs.n2n_construct(G)
    new = graphs.n2n_construct(G)
    np.testing.assert_equal(old.todense(), new.todense())


def collapse_mat(n_edges):
    """Matrix that combines the rows for each edge.

    ::

       >>> collapse_mat(3).todense()
       matrix([[1., 1., 0., 0., 0., 0.],
               [0., 0., 1., 1., 0., 0.],
               [0., 0., 0., 0., 1., 1.]], dtype=float32)

    """
    return sp.sparse.coo_matrix(
        (
            np.ones(n_edges * 2, dtype=np.float32),
            (np.arange(n_edges * 2) // 2, np.arange(n_edges * 2)),
        ),
        shape=(n_edges, n_edges * 2),
    ).tocsr()


@given(graph_builder(graph_type=nx.Graph, min_nodes=2, min_edges=1))
def test_e2n_construct(G):
    note(str(nx.nx_pydot.to_pydot(G)))
    old = orig_graphs.e2n_construct(G) @ collapse_mat(len(G.edges)).transpose()
    new = graphs.e2n_construct(G)
    np.testing.assert_equal(old.todense(), new.todense())


@given(graph_builder(graph_type=nx.Graph, min_nodes=2, min_edges=1))
def test_n2e_construct(G):
    note(str(nx.nx_pydot.to_pydot(G)))
    old = collapse_mat(len(G.edges)) @ orig_graphs.n2e_construct(G)
    new = graphs.n2e_construct(G)
    np.testing.assert_equal(old.todense(), new.todense())


@given(graph_builder(graph_type=nx.Graph, min_nodes=2, min_edges=1))
def test_e2e_construct(G):
    note(str(nx.nx_pydot.to_pydot(G)))
    collapse = collapse_mat(len(G.edges))
    old = collapse @ orig_graphs.e2e_construct(G) @ collapse.transpose()
    new = graphs.e2e_construct(G)
    np.testing.assert_equal(old.todense(), new.todense())
