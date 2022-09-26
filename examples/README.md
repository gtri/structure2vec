# Reproducing structure2vec

These examples allow reproducing the experiments
from the original structure2vec paper:
[arXiv:1603.05629](https://arxiv.org/abs/1603.05629) [cs.LG].

Since the reproduction needs RDKit,
an Anaconda environment is used.
Anaconda is designed around publishing things,
so there's a little bit of fiddling around necessary to get this to work.

From the top of this repo:

``` sh
# If you already have an environment with conda-build,
# you can just activate it instead of creating a new one.
conda create -n conda-build
conda activate conda-build
conda install conda-build conda-verify
conda build -c pytorch .
# conda build will print the path to the package
# which will end with something like
# conda-bld/noarch/structure2vec-*.tar.bz2
REPO="path/to/conda-bld"
conda create -n s2v-examples
conda activate s2v-examples
conda install -c "file://$REPO" -c rdkit -c pytorch install \
    structure2vec rdkit ignite pytorch click
```

At this point,
you should be able to run the example scripts.
They have built-in help messages describing the options.
