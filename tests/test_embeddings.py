import networkx as nx
import numpy as np
import pytest
import torch

from structure2vec.discriminative_embedding import (
    LoopyBeliefPropagation,
    MeanFieldInference,
)
from structure2vec.graph_collate import Graph, graph_collate

embedding_size = 4


@pytest.mark.parametrize("link", [MeanFieldInference, LoopyBeliefPropagation])
@pytest.mark.parametrize("has_edge_features", [False, True])
@pytest.mark.parametrize(
    "recursive", [None, torch.nn.Linear(embedding_size, embedding_size)]
)
def test_embedding(link, has_edge_features, recursive):
    """Check basic type coherence with various options.

    The invariant being checked here is that the embedding returns a
    tensor of the expected size, i.e. a batch dimension equal to the
    number of batches and a features dimension with the specified
    embedding size.

    """
    graphs = [nx.fast_gnp_random_graph(n, 0.25) for n in (10, 11, 12)]

    node_features = 6
    node_feature_mats = [
        np.random.normal(size=(len(g.nodes), node_features)).astype(np.float32)
        for g in graphs
    ]

    if has_edge_features:
        edge_features = 8
        edge_feature_mats = [
            np.random.normal(size=(len(g.edges), edge_features)).astype(np.float32)
            for g in graphs
        ]
        feature_mats = list(zip(node_feature_mats, edge_feature_mats))
    else:
        edge_features = 0
        feature_mats = list(zip(node_feature_mats))

    embedding = link(node_features, edge_features, embedding_size, recursive=recursive)

    graph_structures = graph_collate(
        [Graph(embedding.graph_structure(g), f) for g, f in zip(graphs, feature_mats)]
    )

    embeddings = embedding(graph_structures)

    assert embeddings.shape == (len(graphs), embedding_size)
