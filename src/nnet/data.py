import gzip
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import jax.numpy as jnp
import numpy as np
from jax.nn import one_hot
from jaxlib.xla_extension import DeviceArray
from nnet.config import BLD
from nnet.config import NUM_CLASSES


# ======================================================================================
# Simulation of regression model
# ======================================================================================


def simulate_data(
    n_samples,
    *,
    n_features=None,
    n_informative=None,
    noise=0.5,
    nonlinear=False,
    seed=0,
    return_coef=False,
):
    """Simulate regression data.

    Args:
        n_samples (int): The number of samples.
        n_features (int): The number of features. If None, the number is computed using
            n_features = np.floor(0.1 * n_samples)
        n_informative (int): The number of informative features, i.e., the number of
            features used to build the linear model used to generate the output. If
            None, is set to np.floor(0.1 * n_features).
        noise (float): The standard deviation of the gaussian noise applied to the
            output. Default is 0.5.
        nonlinear (bool): Whether X influences Y through a nonlinear mapping.
        seed (int): Random state initializer passed to `np.random.default_rng`. Default
            is 0.
        return_coef (bool): Whether the true coefficients should be returned.

    Returns:
        - X: np.ndarray of shape (n_samples, n_features). The input samples.
        - y: np.ndarray of shape (n_samples,). The output values.
        - coef: np.ndarray of shape (n_features,). The coefficient of the underlying
        linear model.

    """
    rng = np.random.default_rng(seed=seed)

    if n_features is None:
        n_features = int(np.floor(0.1 * n_samples))

    if n_informative is None:
        n_informative = int(np.floor(0.1 * n_features))

    X = rng.normal(size=(n_samples, n_features))
    X_informative = X[:, :n_informative]
    noise = rng.normal(size=n_samples, scale=noise)
    coef = simulate_coefficients(n_params=n_informative)

    if nonlinear:
        y = nonlinear_mapping(X_informative, coef) + noise
    else:
        y = X_informative @ coef + noise

    if return_coef:
        out = X, y, coef
    else:
        out = X, y

    return out


def simulate_coefficients(n_params):
    rng = np.random.default_rng(seed=n_params)
    coef = rng.integers(low=-1_000, high=1_000, size=n_params)
    coef = coef / 100
    coef = np.round(coef, decimals=2)
    return coef


def nonlinear_mapping(x, coef):
    z = (x**2) @ coef
    return (np.sin(z) + np.clip(z, a_min=0, a_max=None)) / 100


# ======================================================================================
# Loading of MNIST data set
# ======================================================================================


def get_mnist_data(path=None):
    """Fetch training and testing mnist data.

    This function is inspired by
    https://github.com/RobertTLange/code-and-blog/tree/master/04_jax_intro.

    Args:
        path (str): Directory containing the data. Create if nonexistent. Download any
            missing files.

    Returns:
        namedtuple with keys 'test' and 'train'. Each entry is itself a namedtuple with
        the following entries:
        - images: DeviceArray[float] of shape (n_samples, 28 ** 2)
        - labels: DeviceArray[int] of shape (n_samples, )
        - onehot: DeviceArray[int] of shape (n_samples, 10)
        - predictions: DeviceArray[float] of shape (n_samples, 10)

    """
    ####################################################################################
    # create data directory (if not exists)
    ####################################################################################

    if path is None:
        path = BLD.joinpath("data")
    else:
        path = Path(path)

    path.mkdir(parents=True, exist_ok=True)

    ####################################################################################
    # download data
    ####################################################################################

    # source
    url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        "train-images": "train-images-idx3-ubyte.gz",
        "train-labels": "train-labels-idx1-ubyte.gz",
        "test-images": "t10k-images-idx3-ubyte.gz",
        "test-labels": "t10k-labels-idx1-ubyte.gz",
    }

    # download (missing) files
    for file in files.values():
        if file not in (f.name for f in path.iterdir()):
            urlretrieve(url + file, path.joinpath(file))
            warnings.warn(f"Downloaded {file} to {path}")

    ####################################################################################
    # extract data
    ####################################################################################

    def _get_images(path):
        """Return images (features) loaded locally."""
        with gzip.open(path) as f:
            # first 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = jnp.frombuffer(f.read(), "B", offset=16)
        images = pixels.reshape(-1, 784).astype(jnp.float32) / 255
        return images

    def _get_labels(path):
        """Return labels (outcomes) loaded locally."""
        with gzip.open(path) as f:
            # first 8 bytes are magic_number, n_labels
            integer_labels = jnp.frombuffer(f.read(), "B", offset=8)

        return integer_labels

    train_images = _get_images(path.joinpath(files["train-images"]))
    train_labels = _get_labels(path.joinpath(files["train-labels"]))

    test_images = _get_images(path.joinpath(files["test-images"]))
    test_labels = _get_labels(path.joinpath(files["test-labels"]))

    ####################################################################################
    # standardize features
    ####################################################################################

    mean = train_images.mean()
    std = train_images.std()

    train_images -= mean
    train_images /= std

    test_images -= mean
    test_images /= std

    ####################################################################################
    # combine output
    ####################################################################################

    data = TrainTestContainer(
        train=Data(
            images=train_images,
            labels=train_labels,
            onehot=one_hot(train_labels, num_classes=NUM_CLASSES),
        ),
        test=Data(
            images=test_images,
            labels=test_labels,
            onehot=one_hot(test_labels, num_classes=NUM_CLASSES),
        ),
    )
    return data


def get_batch(batch_id, data, *, batch_size):
    """Extract a batch from data.

    Args:
        batch_id (int): The batch id.
        data (Data): The data container.
        batch_size (int): Size of each batch.

    Returns:
        tuple: (images, labels)

    """
    _from = batch_id * batch_size
    _to = (batch_id + 1) * batch_size
    return data.images[_from:_to], data.onehot[_from:_to]


@dataclass
class Data:
    images: DeviceArray = None
    labels: DeviceArray = None
    onehot: DeviceArray = None
    predictions: DeviceArray = None


@dataclass
class TrainTestContainer:
    train: object
    test: object
