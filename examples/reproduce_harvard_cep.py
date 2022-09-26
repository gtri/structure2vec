import contextlib
import datetime
import functools
import json
import sys
import time

import click
import ignite
import networkx as nx
import numpy as np
import scipy as sp
import torch
from ignite.engine import Events

from structure2vec.discriminative_embedding import (
    LoopyBeliefPropagation,
    MeanFieldInference,
)
from structure2vec.graph_collate import Graph, graph_collate


def log_metrics(engine):
    engine.logger.info(
        "Epoch[%s] metrics: %s", engine.state.epoch, engine.state.metrics
    )


@contextlib.contextmanager
def timer(event):
    print(f"{event.capitalize()}...", file=sys.stderr)
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    delta = datetime.timedelta(seconds=end - start)
    print(f"Done {event}. {delta}", file=sys.stderr)


def load_graphs(fp):
    for line in fp:
        yield nx.readwrite.json_graph.node_link_graph(json.loads(line))


class GraphFeatures:
    def __init__(self):
        self.rows = []
        self.cols = []
        self.data = []
        self._row = 0
        self._col = 0

    def add_categorical(self, index, size):
        if index < size:
            self.rows.append(self._row)
            self.cols.append(self._col + index)
            self.data.append(1.0)
        self._col += size

    def add_feature(self, value):
        self.rows.append(self._row)
        self.cols.append(self._col)
        self.data.append(value)
        self._col += 1

    def next_row(self):
        self._row += 1
        self._col = 0

    def build(self):
        rows = np.array(self.rows)
        cols = np.array(self.cols)
        return sp.sparse.coo_matrix(
            (np.array(self.data, dtype=np.float32), (rows, cols)),
            shape=(rows.max() + 1, cols.max() + 1),
        )


def node_feature_count(counts):
    return (
        counts["atomic_number"]
        + counts["degree"]
        + counts["n_hydrogen"]
        + counts["valence"]
        + 1
    )


def extract_node_features(
    row, data, max_atomic_number, max_degree, max_n_hydrogen, max_valence
):
    row.add_categorical(data["atomic_number"], max_atomic_number)
    row.add_categorical(data["degree"], max_degree)
    row.add_categorical(data["n_hydrogen"], max_n_hydrogen)
    row.add_categorical(data["valence"], max_valence)
    row.add_feature(float(data["aromatic"]))
    row.next_row()


def node_features(graph, set_row):
    features = GraphFeatures()
    for _n, d in graph.nodes(data=True):
        set_row(features, d)
    return features.build().todense()


def edge_feature_count(counts):
    return counts["bond_type"] + 2


def extract_edge_features(row, data, max_bond_type):
    row.add_categorical(data["bond_type"], max_bond_type)
    row.add_feature(float(data["conjugated"]))
    row.add_feature(float(data["ring"]))
    row.next_row()


def edge_features(graph, set_row):
    features = GraphFeatures()
    for _u, _v, d in graph.edges(data=True):
        set_row(features, d)
    return features.build().todense()


def graphs_to_dataset(graphs, extract_structure, extract_features):
    for G in graphs:
        yield Graph(
            structure=extract_structure(G), features=[f(G) for f in extract_features]
        ), graph_label(G)


def graph_label(G):
    return np.array(G.graph["label"], dtype=np.float32)


class DiscriminativeEmbedding(torch.nn.Module):
    def __init__(self, embed_link, embedding_dimension, hidden_nodes, hidden_depth):
        super().__init__()
        steps = [embed_link]
        in_size = embedding_dimension
        for _ in range(hidden_depth):
            steps.extend((torch.nn.Linear(in_size, hidden_nodes), torch.nn.ReLU()))
            in_size = hidden_nodes
        steps.append(torch.nn.Linear(in_size, 1))
        self.model = torch.nn.Sequential(*steps)

    def forward(self, args):
        return torch.squeeze(self.model(args))


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
    help="Number of iterations of the embedding algorithm.",
)
@click.option(
    "--embedding-dimension",
    type=click.IntRange(1, None),
    default=64,
    help="Number of dimensions to use for the embedding.",
)
@click.option(
    "--hidden-nodes",
    type=click.IntRange(1, None),
    default=64,
    help="Number of hidden nodes to use for DNN layers.",
)
@click.option(
    "--hidden-depth",
    type=click.IntRange(1, None),
    default=2,
    help="Number of DNN layers.",
)
@click.option(
    "--epochs",
    type=click.IntRange(1, None),
    default=10000,
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
@click.argument("train_dataset", type=click.File("rt"))
@click.argument("test_dataset", type=click.File("rt"))
@click.argument("categorical_counts", type=click.File("rt"))
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
    train_dataset,
    test_dataset,
    categorical_counts,
    output_dir,
):
    categorical_counts = json.load(categorical_counts)
    set_node_row = functools.partial(
        extract_node_features,
        max_atomic_number=categorical_counts["atomic_number"],
        max_degree=categorical_counts["degree"],
        max_n_hydrogen=categorical_counts["n_hydrogen"],
        max_valence=categorical_counts["valence"],
    )
    set_edge_row = functools.partial(
        extract_edge_features, max_bond_type=categorical_counts["bond_type"]
    )
    n_node_features = node_feature_count(categorical_counts)
    n_edge_features = edge_feature_count(categorical_counts)

    if embedding == "mean-field":
        embed_link = MeanFieldInference(
            n_node_features,
            n_edge_features,
            embedding_dimension,
            steps=embedding_iterations,
        )
    elif embedding == "loopy-belief-propagation":
        embed_link = LoopyBeliefPropagation(
            n_node_features,
            n_edge_features,
            embedding_dimension,
            steps=embedding_iterations,
        )

    with timer("loading graphs"):
        graph_column_functions = [
            functools.partial(node_features, set_row=set_node_row),
            functools.partial(edge_features, set_row=set_edge_row),
        ]
        train_dataset = list(
            graphs_to_dataset(
                load_graphs(train_dataset),
                embed_link.graph_structure,
                graph_column_functions,
            )
        )
        test_dataset = list(
            graphs_to_dataset(
                load_graphs(test_dataset),
                embed_link.graph_structure,
                graph_column_functions,
            )
        )

    with timer("setting up model"):
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=graph_collate,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=graph_collate,
        )

        model = DiscriminativeEmbedding(
            embed_link, embedding_dimension, hidden_nodes, hidden_depth
        ).to(device=device)
        optimizer = torch.optim.Adam(model.parameters())
        loss = torch.nn.MSELoss()
        trainer = ignite.engine.create_supervised_trainer(
            model, optimizer, loss, device=device
        )
        evaluator = ignite.engine.create_supervised_evaluator(
            model,
            metrics={
                "mae": ignite.metrics.MeanAbsoluteError(),
                "mse": ignite.metrics.MeanSquaredError(),
            },
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
                score_function=lambda e: -e.state.metrics["mse"],
                trainer=trainer,
                min_delta=1e-6,
            ),
        )
        evaluator.add_event_handler(Events.COMPLETED, log_metrics)

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == "__main__":
    main()
