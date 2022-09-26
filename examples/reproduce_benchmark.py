"""This script contains code for reproducing benchmarks.

The benchmark experiments are described in Section 6.1 of the
structure2vec paper.

"""
import collections
import functools
import itertools

import click
import ignite
import networkx as nx
import numpy as np
import torch
from ignite.engine import Events

from structure2vec.discriminative_embedding import (
    LoopyBeliefPropagation,
    MeanFieldInference,
)
from structure2vec.graph_collate import Graph, graph_collate

GraphsInfo = collections.namedtuple("GraphsInfo", "graphs n_labels n_tags")


def log_metrics(engine):
    engine.logger.info(
        "Epoch[%s] metrics: %s", engine.state.epoch, engine.state.metrics
    )


def load_graphs(fh):
    """Load graphs from the extended adjacency format.

    Pass a file handle (really any iterable of strings). The format is
    as described in the original structure2vec repository's data `README
    <https://github.com/Hanjun-Dai/pytorch_structure2vec/blob/master/graph_classification/data/README.md>`_.

    .. epigraph::

        * 1st line: ``N`` number of graphs; then the following ``N``
          blocks describe the graphs
        * for each block of text:
            - a line contains ``n l``, where ``n`` is number of nodes in
              the current graph, and ``l`` is the graph label
            - following ``n`` lines:
                - the ``i``th line describes the information of ``i``th
                  node (0 based), which starts with ``t m``, where ``t``
                  is the tag of current node, and ``m`` is the number of
                  neighbors of current node;
                - following ``m`` numbers indicate the neighbor indices
                  (starting from 0).

    Three items are returned in a tuple:
    * List of :cls:`networkx.Graph`.
    * Number of distinct labels.
    * Number of distinct node tags.

    """
    rows = ([int(x) for x in l.split()] for l in fh)

    graphs = []
    label_translation = collections.defaultdict(itertools.count().__next__)
    tag_translation = collections.defaultdict(itertools.count().__next__)

    (graph_count,) = next(rows)
    for _ in range(graph_count):
        node_count, graph_label = next(rows)
        g = nx.Graph(label=label_translation[graph_label])
        for i in range(node_count):
            node_tag, _node_degree, *neighbors = next(rows)
            g.add_node(i, tag=tag_translation[node_tag])
            for neighbor in neighbors:
                g.add_edge(i, neighbor)
        graphs.append(g)
    return GraphsInfo(graphs, len(label_translation), len(tag_translation))


def graphs_to_dataset(graphs, structure_extractor, feature_extractor):
    for G in graphs:
        yield (
            Graph(structure=structure_extractor(G), features=(feature_extractor(G),)),
            graph_label(G),
        )


def node_features(G, n_features):
    return np.eye(n_features, dtype=np.float32)[
        np.array([t for n, t in G.nodes(data="tag")])
    ]


def graph_label(G):
    return G.graph["label"]


@click.command()
@click.option(
    "--embedding",
    type=click.Choice(["mean-field", "loopy-belief-propagation"]),
    default="mean-field",
    show_default=True,
    help="Embedding to use",
)
@click.option(
    "--embedding-iterations",
    type=click.IntRange(1, None),
    default=3,
    show_default=True,
    help="Number of iterations of the embedding algorithm.",
)
@click.option(
    "--embedding-dimension",
    type=click.IntRange(1, None),
    default=64,
    show_default=True,
    help="Number of dimensions to use for the embedding.",
)
@click.option(
    "--hidden-nodes",
    type=click.IntRange(1, None),
    default=64,
    show_default=True,
    help="Number of hidden nodes to use for DNN layers.",
)
@click.option(
    "--hidden-depth",
    type=click.IntRange(1, None),
    default=2,
    show_default=True,
    help="Depth of DNN.",
)
@click.option(
    "--epochs",
    type=click.IntRange(1, None),
    default=10000,
    show_default=True,
    help="Number of training epochs.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(1, None),
    default=50,
    show_default=True,
    help="Training batch size.",
)
@click.option(
    "--device",
    type=torch.device,
    default="cuda",
    show_default=True,
    help="Device to use.",
)
@click.argument("graph_file", type=click.File("rt"))
@click.argument("train_indexes", type=click.File("rt"))
@click.argument("test_indexes", type=click.File("rt"))
@click.argument("output_dir", type=click.Path(file_okay=False, writable=True))
def main(
    embedding,
    embedding_iterations,
    embedding_dimension,
    hidden_nodes,
    hidden_depth,
    epochs,
    batch_size,
    device,
    graph_file,
    train_indexes,
    test_indexes,
    output_dir,
):
    graphs, n_labels, n_features = load_graphs(graph_file)
    train_graphs = [graphs[int(i.strip())] for i in train_indexes]
    test_graphs = [graphs[int(i.strip())] for i in test_indexes]

    if embedding == "mean-field":
        embed_link = MeanFieldInference(
            n_features, 0, embedding_dimension, steps=embedding_iterations
        )
    elif embedding == "loopy-belief-propagation":
        embed_link = LoopyBeliefPropagation(
            n_features, 0, embedding_dimension, steps=embedding_iterations
        )

    train_dataset = list(
        graphs_to_dataset(
            train_graphs,
            embed_link.graph_structure,
            functools.partial(node_features, n_features=n_features),
        )
    )
    test_dataset = list(
        graphs_to_dataset(
            test_graphs,
            embed_link.graph_structure,
            functools.partial(node_features, n_features=n_features),
        )
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=graph_collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=graph_collate
    )

    steps = [embed_link]
    in_size = embedding_dimension
    for _ in range(hidden_depth):
        steps.extend((torch.nn.Linear(in_size, hidden_nodes), torch.nn.ReLU()))
        in_size = hidden_nodes
    steps.append(torch.nn.Linear(in_size, n_labels))
    model = torch.nn.Sequential(*steps).to(device=device)

    optimizer = torch.optim.Adam(model.parameters())
    loss = torch.nn.CrossEntropyLoss()
    trainer = ignite.engine.create_supervised_trainer(
        model, optimizer, loss, device=device
    )
    evaluator = ignite.engine.create_supervised_evaluator(
        model,
        metrics={"acc": ignite.metrics.Accuracy(), "loss": ignite.metrics.Loss(loss)},
        device=device,
    )

    trainer.logger = ignite.utils.setup_logger("trainer")
    evaluator.logger = ignite.utils.setup_logger("evaluator")
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        ignite.handlers.ModelCheckpoint(output_dir, "model"),
        {"model": model},
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(trainer):
        evaluator.run(test_loader)

    evaluator.add_event_handler(
        Events.COMPLETED,
        ignite.handlers.EarlyStopping(
            patience=5,
            score_function=lambda e: -e.state.metrics["loss"],
            trainer=trainer,
            min_delta=1e-6,
        ),
    )
    evaluator.add_event_handler(Events.COMPLETED, log_metrics)

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == "__main__":
    main()
