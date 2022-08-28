import jax.numpy as jnp
import pytest
from jax.nn import one_hot
from nnet.data import Data
from nnet.data import TrainTestContainer
from nnet.network import _cross_entropy
from nnet.network import _mean_square
from nnet.network import build_network
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@pytest.mark.unit
def test_mean_square():
    pred = jnp.array([1.0, 0, 0, 1, 0, 1])
    onehot = 1 - pred
    got = _mean_square(pred, onehot)
    assert got == len(pred)


@pytest.mark.unit
def test_cross_entropy():
    pred = jnp.exp(jnp.array([1.0, 0, 0, 1, 0, 1]))
    onehot = jnp.ones_like(pred)
    got = _cross_entropy(pred, onehot)
    assert got == -3.0


@pytest.mark.end_to_end
def test_classification_problem():
    """Test that accuracy of a simple network is higher than 96% on a simple problem."""
    x, y = make_classification(
        n_samples=50_000, n_features=10, n_informative=3, n_classes=2, class_sep=2
    )

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    data = TrainTestContainer(
        train=Data(
            images=x_train, labels=y_train, onehot=one_hot(y_train, num_classes=2)
        ),
        test=Data(images=x_test, labels=y_test, onehot=one_hot(y_test, num_classes=2)),
    )

    network = build_network(
        structure=[10, 10, 2],
        loss_type="cross_entropy",
        activation_type="sigmoid",
        algorithm="adam",
        problem="classification",
        step_size=0.001,
    )

    result = network.fit(
        data, n_epochs=100, batch_size=100, tol=1e-7, show_progress=False
    )

    assert result.log.test[-1] > 0.96
