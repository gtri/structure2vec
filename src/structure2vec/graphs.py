import networkx as nx
import numpy as np
import scipy.sparse

# This function imported from NetworkX 2.4 and modified to allow specifying dtype and format.
# Also modified to directly construct a coo_matrix.
# lil_matrix.tocoo is implemented as self.tocsr().tocoo()
# NetworkX is 3-clause BSD; see LICENSE.networkx in the root of this repository.


def _incidence_matrix(  # noqa: C901
    G,
    nodelist=None,
    edgelist=None,
    oriented=False,
    weight=None,
    dtype=np.float_,
    format="csc",
):  # pragma: no cover
    """Returns incidence matrix of G.

    The incidence matrix assigns each row to a node and each column to an edge.
    For a standard incidence matrix a 1 appears wherever a row's node is
    incident on the column's edge.  For an oriented incidence matrix each
    edge is assigned an orientation (arbitrarily for undirected and aligning to
    direction for directed).  A -1 appears for the tail of an edge and 1
    for the head of the edge.  The elements are zero otherwise.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list, optional   (default= all nodes in G)
       The rows are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    edgelist : list, optional (default= all edges in G)
       The columns are ordered according to the edges in edgelist.
       If edgelist is None, then the ordering is produced by G.edges().

    oriented: bool, optional (default=False)
       If True, matrix elements are +1 or -1 for the head or tail node
       respectively of each edge.  If False, +1 occurs at both nodes.

    weight : string or None, optional (default=None)
       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.  Edge weights, if used,
       should be positive so that the orientation can provide the sign.

    Returns
    -------
    A : SciPy sparse matrix
      The incidence matrix of G.

    Notes
    -----
    For MultiGraph/MultiDiGraph, the edges in edgelist should be
    (u,v,key) 3-tuples.

    "Networks are the best discrete model for so many problems in
    applied mathematics" [1]_.

    References
    ----------
    .. [1] Gil Strang, Network applications: A = incidence matrix,
       http://academicearth.org/lectures/network-applications-incidence-matrix
    """
    if nodelist is None:
        nodelist = list(G)
    if edgelist is None:
        if G.is_multigraph():
            edgelist = list(G.edges(keys=True))
        else:
            edgelist = list(G.edges())
    rows = []
    cols = []
    data = []
    node_index = {node: i for i, node in enumerate(nodelist)}
    for ei, e in enumerate(edgelist):
        (u, v) = e[:2]
        if u == v:
            continue  # self loops give zero column
        try:
            ui = node_index[u]
            vi = node_index[v]
        except KeyError:
            raise nx.NetworkXError(f"node {u} or {v} in edgelist but not in nodelist")
        if weight is None:
            wt = 1
        else:
            if G.is_multigraph():
                ekey = e[2]
                wt = G[u][v][ekey].get(weight, 1)
            else:
                wt = G[u][v].get(weight, 1)
        if oriented:
            rows.append(ui)
            cols.append(ei)
            data.append(-wt)
            rows.append(vi)
            cols.append(ei)
            data.append(wt)
        else:
            rows.append(ui)
            cols.append(ei)
            data.append(wt)
            rows.append(vi)
            cols.append(ei)
            data.append(wt)
    A = scipy.sparse.coo_matrix(
        (np.array(data, dtype=dtype), (np.array(rows), np.array(cols))),
        shape=(len(nodelist), len(edgelist)),
    )
    return A.asformat(format)


# End imported code


def n2n_construct(G):
    return nx.to_scipy_sparse_matrix(G, weight=None, dtype=np.float32, format="coo")


def e2n_construct(G):
    return _incidence_matrix(G, weight=None, dtype=np.float32, format="coo")


def n2e_construct(G):
    return _incidence_matrix(G, weight=None, dtype=np.float32, format="coo").transpose()


def e2e_construct(G):
    return nx.to_scipy_sparse_matrix(
        nx.line_graph(G),
        nodelist=[tuple(sorted(x)) for x in G.edges],
        weight=None,
        dtype=np.float32,
        format="coo",
    )
