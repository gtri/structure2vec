import torch

from structure2vec.graphs import (
    e2e_construct,
    e2n_construct,
    n2e_construct,
    n2n_construct,
)


class MeanFieldInference(torch.nn.Module):
    """Mean Field Inference

    This learnable link performs the Mean Field Inference dimensional
    embedding. The shape of the data set is determined by whether or not
    edges have features. If edges have no features, then the
    :class:`~structure2vec.graph_collate.Graph` should be::

        Graph(structure=(n2n_construct(g),), features=(node_features,))

    If edges have features, then it should be::

        Graph(
            structure=(n2n_construct(g), e2n_construct(g)),
            features=(node_features, edge_features),
        )

    Node features should be in the standard NetworkX traversal order.
    Edge features should be in the standard NetworkX traversal order.

    :param int out_size: The number of dimensions in the latent space.
    :param int steps:
       The number of iterations of the approximation to run.
    :param activation: A stateless activation function.
    :param recursive:
       The link to use for the recursive update. If omitted, defaults to
       ``torch.nn.Linear(None, out_size, bias=False)``.

    """

    def __init__(
        self,
        node_features,
        edge_features,
        out_size,
        *,
        steps=3,
        activation=torch.nn.functional.relu,
        recursive=None,
    ):
        super().__init__()
        self.steps = steps
        self.activation = activation

        if recursive is None:
            recursive = torch.nn.Linear(out_size, out_size, bias=False)

        self.n2l = torch.nn.Linear(node_features, out_size, bias=False)
        if edge_features:
            self.e2l = torch.nn.Linear(edge_features, out_size, bias=False)
        else:
            self.e2l = self.add_module("e2l", None)
        self.recursive = recursive

    def forward(self, graph):
        if self.e2l is None:
            reduction_sp, n2n_sp = graph.structure
            (node_feat,) = graph.features
        else:
            reduction_sp, n2n_sp, e2n_sp = graph.structure
            node_feat, edge_feat = graph.features

        input_message = self.n2l(node_feat)
        if self.e2l is not None:
            input_message += torch.sparse.mm(e2n_sp, self.e2l(edge_feat))

        message = self.activation(input_message)
        for _ in range(self.steps):
            message = self.activation(
                input_message + self.recursive(torch.sparse.mm(n2n_sp, message))
            )
        return torch.sparse.mm(reduction_sp, message)

    def graph_structure(self, graph):
        if self.e2l is None:
            return (n2n_construct(graph),)
        else:
            return (n2n_construct(graph), e2n_construct(graph))


class LoopyBeliefPropagation(torch.nn.Module):
    """Loopy Belief Propagation

    This learnable link performs the Loopy Belief Propagation dimensional
    embedding. The shape of the data set is determined by whether or not
    edges have features. If edges have no features, then the
    :class:`~structure2vec.graph_collate.Graph` should be::

        Graph(
            structure=(e2e_construct(g), n2e_construct(g)),
            features=(node_features,),
        )

    If edges have features, then it should be::

        Graph(
            structure=(e2e_construct(g), n2e_construct(g)),
            features=(node_features, edge_features),
        )

    Node features should be in the standard NetworkX traversal order.
    Edge features should be in the standard NetworkX traversal order.

    :param int out_size: The number of dimensions in the latent space.
    :param int steps:
       The number of iterations of the approximation to run.
    :param activation: A stateless activation function.
    :param recursive:
       The link to use for the recursive update. If omitted, defaults to
       ``torch.nn.Linear(None, out_size, bias=False)``.

    """

    def __init__(
        self,
        node_features,
        edge_features,
        out_size,
        *,
        steps=3,
        activation=torch.nn.functional.relu,
        recursive=None,
    ):
        super().__init__()
        self.steps = steps
        self.activation = activation

        if recursive is None:
            recursive = torch.nn.Linear(out_size, out_size, bias=False)

        self.n2l = torch.nn.Linear(node_features, out_size, bias=False)
        if edge_features:
            self.e2l = torch.nn.Linear(edge_features, out_size, bias=False)
        else:
            self.e2l = self.add_module("e2l", None)
        self.recursive = recursive

    def forward(self, graph):
        reduction_sp, e2e_sp, n2e_sp = graph.structure
        if self.e2l is None:
            (node_feat,) = graph.features
        else:
            node_feat, edge_feat = graph.features

        input_message = torch.sparse.mm(n2e_sp, self.n2l(node_feat))
        if self.e2l is not None:
            input_message += self.e2l(edge_feat)

        message = self.activation(input_message)
        for _ in range(self.steps):
            message = self.activation(
                input_message + self.recursive(torch.sparse.mm(e2e_sp, message))
            )
        return torch.sparse.mm(reduction_sp, message)

    def graph_structure(self, graph):
        return (e2e_construct(graph), n2e_construct(graph))
