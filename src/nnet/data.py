import gzip
import warnings
from pathlib import Path
from typing import NamedTuple
from urllib.request import urlretrieve

import jax.numpy as jnp
from jax.nn import one_hot
from jaxlib.xla_extension import DeviceArray
from nnet.config import BLD
from nnet.config import NUM_CLASSES


def get_batch(batch_id, data, *, batch_size):
    """Extract batch from data.

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


def get_mnist_data(path=None):
    """Fetch training and testing mnist data.

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


class Data(NamedTuple):
    """Data container."""

    images: DeviceArray = None
    labels: DeviceArray = None
    onehot: DeviceArray = None
    predictions: DeviceArray = None


class TrainTestContainer(NamedTuple):
    """Train/test container."""

    train: object
    test: object
