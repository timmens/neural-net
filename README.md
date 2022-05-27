# Neural Networks

[![image](https://img.shields.io/github/workflow/status/timmens/nnet/main/main)](https://github.com/timmens/nnet/actions?query=branch%3Amain)
[![image](https://readthedocs.org/projects/nnet/badge/?version=stable)](https://nnet.readthedocs.io/en/stable/?badge=stable)
[![image](https://codecov.io/gh/timmens/nnet/branch/main/graph/badge.svg)](https://codecov.io/gh/timmens/nnet)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/timmens/nnet/main.svg)](https://results.pre-commit.ci/latest/github/timmens/nnet/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project contains an implementation of a vanilla MLP neural network using JAX. The
network is trained on the MNIST data set to recognize hand-written digits.

### Topics in Econometrics and Statistics

> Summer 2020, Tim Mensinger

## Usage

To get started, create and activate the environment with

```bash
$ cd 'into project root'
$ conda/mamba env create -f environment.yml
$ conda activate nnet
```

Now you can run the relevant notebooks.

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[cookiecutter-pytask-project](https://github.com/pytask-dev/cookiecutter-pytask-project)
template.
