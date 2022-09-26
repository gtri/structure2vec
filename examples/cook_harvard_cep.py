"""This converts the SMILES data to NetworkX.

This happens in a separate script since RDKit is distributed only via
Anaconda and Chainer is distributed only via PyPI. Sigh.

    conda install -c rdkit rdkit
    conda install click networkx

"""
import collections
import itertools
import json

import click
import networkx as nx
from rdkit import Chem


def molecule_to_graph(mol, categories):
    G = nx.Graph(label=float(mol.GetProp("_Name")))
    ring_info = mol.GetRingInfo()
    for a in mol.GetAtoms():
        G.add_node(
            a.GetIdx(),
            atomic_number=categories["atomic_number"][a.GetAtomicNum()],
            degree=categories["degree"][a.GetDegree()],
            n_hydrogen=categories["n_hydrogen"][a.GetTotalNumHs()],
            valence=categories["valence"][a.GetImplicitValence()],
            aromatic=a.GetIsAromatic(),
        )
    for b in mol.GetBonds():
        G.add_edge(
            b.GetBeginAtomIdx(),
            b.GetEndAtomIdx(),
            bond_type=categories["bond_type"][b.GetBondType()],
            conjugated=b.GetIsConjugated(),
            ring=bool(ring_info.NumBondRings(b.GetIdx())),
        )
    return G


@click.command()
@click.option(
    "--category-counts",
    type=click.Path(dir_okay=False, writable=True),
    default="category_counts.json",
    show_default=True,
    help="Path to file where categorical counts will be written.",
)
@click.argument("input_files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def main(category_counts, input_files):
    categories = collections.defaultdict(
        lambda: collections.defaultdict(itertools.count().__next__)
    )
    for input_file in input_files:
        with open(input_file + ".ndjson", "w") as output_file:
            for mol in Chem.SmilesMolSupplier(input_file):
                G = molecule_to_graph(mol, categories)
                print(
                    json.dumps(nx.readwrite.json_graph.node_link_data(G)),
                    file=output_file,
                )

    with open(category_counts, "w") as f:
        json.dump({k: len(v) for k, v in categories.items()}, f, indent=2)


if __name__ == "__main__":
    main()
