# structure2vec

Pure PyTorch implementation of structure2vec.
The structure2vec algorithm is described in

Dai, Hanjun, Dai, Bo, and Song, Le.
"Discriminative Embeddings of Latent Variable Models for Structured Data" 
arXiv preprint arXiv:1603.05629.
Online: https://arxiv.org/abs/1603.05629

with code at https://github.com/Hanjun-Dai/pytorch_structure2vec.

This implementation was used in 

Yisroel Mirsky, George Macon, Michael Brown, Carter Yagemann, Matthew Pruett, Evan Downing, Sukarno Mertoguno, and Wenke Lee.
"VulChecker: Graph-based Vulnerability Localization in Source Code"
USENIX Security 2023.

with code at https://github.com/ymirsky/VulChecker.

# Documentation

There's documentation for this package in the ``docs`` directory.
If you have Tox and Python 3.8 installed,
you can build the documentation by running

    tox -e docs

Then open ``docs/_build/html/index.html`` in your web browser.

# Developer Quickstart

With the project cloned and a virtual environment active:

```bash
pip install -e .[dev,tests,docs]
```

You should configure [pre-commit](https://pre-commit.com/) to check your code before you commit:

```bash
pre-commit install
```

To run the tests, you will need all supported versions of Python installed.
On Ubuntu, you can use the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa).
In other places, you can use [pyenv](https://github.com/pyenv/pyenv).
You can run the automated tests by saying:

```bash
tox
```

# Conda Package

Because we're in data-science-Python here,
there is metadata to build a Conda package of this library in `meta.yaml`.
When updating the version number of this package
or changing the dependencies,
make sure you update both places.

N.B.: The package names for Conda and PyPI may be different
for a particular piece of software.
For example, PyTorch is known as `torch` on PyPI,
but `pytorch` in the `pytorch` Anaconda channel.
