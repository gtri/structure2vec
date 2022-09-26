"""Implementations exactly matching pytorch_structure2vec.

These functions exist to allow chaining together equivalence tests to
see that the functions implemented by structure2vec have the same effect
as the original authors'.

"""

import functools

import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse


def edge_order(G):
    if G.is_directed():
        return list(G.edges)
    else:
        order = []
        for u, v in G.edges:
            order.append((u, v))
            order.append((v, u))
        return order


def _normalize_return(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        r = f(*args, **kwargs)
        r = r.tocoo()
        r.eliminate_zeros()
        r.sum_duplicates()
        return r.astype(np.float32)

    return wrapper


@_normalize_return
def n2n_construct(G):
    return nx.adjacency_matrix(G)


@_normalize_return
def e2n_construct(G):
    VE = nx.incidence_matrix(
        G.to_directed(as_view=True),
        nodelist=list(G),
        edgelist=edge_order(G),
        oriented=True,
    )
    VE[VE == -1] = 0
    return VE


@_normalize_return
def n2e_construct(G):
    EV = nx.incidence_matrix(
        G.to_directed(as_view=True),
        nodelist=list(G),
        edgelist=edge_order(G),
        oriented=True,
    ).transpose()
    EV[EV == 1] = 0
    EV *= -1
    return EV


@_normalize_return
def e2e_construct(G):
    D = G.to_directed(as_view=True)
    edges = edge_order(G)
    result = []
    edge_index = {(u, v): i for i, (u, v) in enumerate(edges)}
    for i, (u, v) in enumerate(edges):
        for w, _ in D.in_edges(u):
            if w == v:
                continue
            j = edge_index[(w, u)]
            result.append((i, j, 1.0))
    if result:
        row, col, data = zip(*result)
    else:
        row = col = data = ()
    return sp.sparse.coo_matrix(
        (np.array(data), (np.array(row), np.array(col))), shape=(len(edges), len(edges))
    )
