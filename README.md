# Neural Networks

[![image](https://img.shields.io/github/workflow/status/timmens/neural-net/main/main)](https://github.com/timmens/neural-net/actions?query=branch%3Amain)
[![image](https://codecov.io/gh/timmens/neural-net/branch/main/graph/badge.svg)](https://codecov.io/gh/timmens/neural-net)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/timmens/neural-net/main.svg)](https://results.pre-commit.ci/latest/github/timmens/neural-net/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### Topics in Econometrics and Statistics

> Summer 2022, Tim Mensinger

This repository contains (source) material for

1. A custom implementation of a vanilla MLP neural network using JAX. The network is
   trained on the MNIST data set to recognize hand-written digits.
1. Two presentations on the theoretical and practical side of neural networks. The
   theoretical version was presented in class.
1. A term paper that analyzes neural networks from an econometric perspective.

## Paper

The tex-files are stored in the folder `[root]/paper` and compiled using (a relatively
fresh install of) latex. You can view it here

- [🗎 Paper](https://github.com/timmens/neural-net/blob/main/paper/main.pdf)

The results in the paper (tables and figures) are generated in a reproducible fashion
using [pytask](www.github.com/pytask-dev). The build process is described below in the
section Code.

## Presentation

There is a *theoretical* and a *practical* presentation. You can view them here

- [🗎 Theoretical](http://htmlpreview.github.io/?https://github.com/timmens/neural-net/blob/main/presentation/theoretical.html)
- [🗎 Practical](http://htmlpreview.github.io/?https://github.com/timmens/neural-net/blob/main/presentation/practical.html)

The source code is stored in the folder `[root]/presentation` and rendered using
[marp](https://marp.app/).

## Code

### Installation

To get started, create and activate the (conda) environment using

```bash
$ cd 'into project root'
$ conda env create -f environment.yml
$ conda activate nnet
```

This installs the package dependencies and the project itself. You can then import
functions defined in the repository as

```python
from nnet.network import build_network
from nnet.plotting import plot_examples
```

### Reproduce Paper

To reproduce the results from the paper (tables and figures) you only have to call
pytask in an active environment

```bash
$ cd 'into project root'
$ conda activate nnet
$ pytask
```

This creates a folder `[root]/bld` which contains all tables and figures used in the
paper in the subfolder `paper`.

### Custom Neural Network Implementation

For this project I implemented a multilayer-perceptron neural network and its fitting
procedure using Jax. The code is found in the module `[root]/src/nnet/network.py`. It
works in both regression and classification settings and is tested on the MNIST dataset
that contains hand-written digits.

> **Warning** Jax is not supported on windows.

If you want to run my custom implementation locally I suggest running the notebook
[`[root]/src/nnet/neural_network.ipynb`](https://github.com/timmens/neural-net/blob/main/src/nnet/classify_digits.ipynb)
---or simply view it on
[nbviewer](https://nbviewer.org/github/timmens/neural-net/blob/main/src/nnet/classify_digits.ipynb).

______________________________________________________________________

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[cookiecutter-pytask-project](https://github.com/pytask-dev/cookiecutter-pytask-project)
template.
