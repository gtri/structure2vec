Structural Embedding with structure2vec
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Introduction
============

This library contains an implementation of the structure2vec algorithm
described in `arXiv:1603.05629 <https://arxiv.org/abs/1603.05629>`_ [cs.LG].
The two embedding estimates described in the paper are included.

The usage pattern of this library is a little unusual
compared to the rest of PyTorch.
Each "observation" is structured as a graph,
and is represented as a tuple of several tensors
instead of as a single tensor.

If you have :class:`networkx.Graph` objects representing your observations,
you're almost ready to use this library.
You will need to write a function to convert a graph
to an instance of :class:`structure2vec.graph_collate.Graph`.
That's a :class:`collections.namedtuple` with two fields:
``structure`` and ``features``.
Both :class:`~structure2vec.discriminative_embedding.MeanFieldInference`
and :class:`~structure2vec.discriminative_embedding.LoopyBeliefPropagation`
have a method named ``graph_structure`` that converts a :class:`networkx.Graph`
into a value suitable for ``structure``.

The ``features`` attribute gets a tuple of :class:`~numpy.ndarray`
for node and edge features,
in that order.
The rows of the node feature matrix correspond to the nodes
in standard traversal order::

   for n in G.nodes:
       ...

The rows of the edge feature matrix correspond to the edges
in standard traversal order::

   for u, v in G.edges:
       ...

It's possible to have no edge features,
in which case you don't provide an edge feature array.
Note that ``features`` must be a tuple even in this case.

In order to use this unusual imput structure with PyTorch,
provide :func:`structure2vec.graph_collate.graph_collate`
as the ``collate_fn`` argument to :class:`torch.utils.data.DataLoader`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
