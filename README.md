# Neural Networks

[![image](https://img.shields.io/github/workflow/status/timmens/neural-net/main/main)](https://github.com/timmens/neural-net/actions?query=branch%3Amain)
[![image](https://codecov.io/gh/timmens/neural-net/branch/main/graph/badge.svg)](https://codecov.io/gh/timmens/neural-net)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/timmens/neural-net/main.svg)](https://results.pre-commit.ci/latest/github/timmens/neural-net/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project contains an implementation of a vanilla MLP neural network using JAX. The
network is trained on the MNIST data set to recognize hand-written digits. It also
contains two presentations on the theoretical and practical side of neural networks.

### Topics in Econometrics and Statistics

> Summer 2022, Tim Mensinger

## Presentation

There is a *theoretical* and a *practical* presentation. You can view them online by
clicking on the respective document icon:

|             |                                                                                                                  |
| ----------- | ---------------------------------------------------------------------------------------------------------------- |
| Theoretical | [🗎](http://htmlpreview.github.io/?https://github.com/timmens/neural-net/blob/main/presentation/theoretical.html) |
| Practical   | [🗎](http://htmlpreview.github.io/?https://github.com/timmens/neural-net/blob/main/presentation/practical.html)   |

I use [marp](https://marp.app/) to convert my markdown files to html presentations. To
get a pdf version simply call

```bash
$ decktape generic -s 1280x720 --load-pause 3000 file.html file.pdf
```

## Usage of the code

> **Warning** Jax is not supported on windows.

To get started, create and activate the environment with

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

or you run the notebook locally ---or view it on
[nbviewer](https://nbviewer.org/github/timmens/neural-net/blob/main/src/nnet/neural_network.ipynb).

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[cookiecutter-pytask-project](https://github.com/pytask-dev/cookiecutter-pytask-project)
template.
