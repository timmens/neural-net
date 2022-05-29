from dataclasses import dataclass
from dataclasses import field
from functools import partial
from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Union

import jax
import jax.numpy as jnp
from jax import jit
from jax import value_and_grad
from jax import vmap
from jax.example_libraries.optimizers import adam
from jax.example_libraries.optimizers import OptimizerState
from jax.example_libraries.optimizers import sgd
from jax.nn import relu
from jax.nn import sigmoid
from jax.nn import softmax
from jaxlib.xla_extension import DeviceArray
from nnet.data import get_batch
from tqdm import tqdm


def build_network(
    *,
    structure,
    loss_type,
    activation_type,
    algorithm,
    tqdm=tqdm,
    step_size=0.01,
    key=None,
):

    # random number generator key
    if key is None:
        key = jax.random.PRNGKey(1)

    # define layer structure
    layer = get_layer(activation_type)

    # get forward pass / predict function
    forward_pass = get_forward_pass_func(layer)

    # get loss function
    loss_func = get_loss_func(forward_pass=forward_pass, loss_type=loss_type)

    # get accuracy function
    compute_accuracy = get_accuracy_func(forward_pass)

    # initialize start parameters
    start_params = get_start_params(structure, key=key)

    # get optimizer triplet
    optimizer = get_optimizer(algorithm=algorithm, step_size=step_size)
    opt_init, opt_update, get_params = optimizer()

    # get update step
    update_params = get_update_func(get_params, opt_update, loss_func)

    # get fitting function
    fit = get_fitting_function(
        update_params=update_params,
        compute_accuracy=compute_accuracy,
        start_params=start_params,
        get_params=get_params,
        opt_init=opt_init,
        tqdm=tqdm,
    )

    # get predict function
    predict = get_predict_func(forward_pass)

    network = Network(
        fit=fit,
        predict=predict,
    )
    return network


def get_predict_func(forward_pass):
    def _predict(params, images):

        predictions = forward_pass(params, images)
        predictions = predictions.argmax(axis=1)

        return predictions

    return _predict


def get_fitting_function(
    update_params,
    compute_accuracy,
    start_params,
    get_params,
    opt_init,
    *,
    tqdm=tqdm,
):
    def _fit(data, *, n_epochs, batch_size, tol=1e-5, show_progress=True):
        """Fit a neural network to data.

        Args:
            data (TrainTestContainer): The training and testing data. See module data.py
            n_epochs (int): Number of epochs.
            batch_size (int): Batch size.
            tol (float): If training accuracy between two epochs is below tol the
                training stops.
            show_progess (bool): Whether to show a progress bar.

        Returns:
            NetworkResults: The network results. Has attributes
            - opt_state (OptimizerState): The optimizer state
            - log (Logging): The log.
            - params: The fitted parameters.

        """
        opt_state = opt_init(start_params)

        ################################################################################
        # initialize logging, parameters and batching
        ################################################################################

        log = Logging()

        params = get_params(opt_state)  # starting parameters that can be overwritten

        n_batches = len(data.train.labels) // batch_size

        ################################################################################
        # create training iterators that show progress
        ################################################################################

        if show_progress:
            epoch_iterator = tqdm(
                range(n_epochs), desc="Training", position=0, leave=True
            )

            def batch_iterator(epoch_id):
                return tqdm(
                    range(n_batches), desc=f"Epoch {epoch_id}", position=1, leave=False
                )

        else:
            epoch_iterator = range(n_epochs)

            def batch_iterator(epoch_id):  # noqa: U100
                return range(n_batches)

        ################################################################################
        # log starting accuracy
        ################################################################################

        train_acc = compute_accuracy(params, data.train)
        test_acc = compute_accuracy(params, data.test)

        log.add_accuracy(train_acc, test_acc)

        ################################################################################
        # training loop
        ################################################################################

        for epoch_id in epoch_iterator:

            for batch_id in batch_iterator(epoch_id):

                batch_image, batch_onehot = get_batch(
                    batch_id, data.train, batch_size=batch_size
                )

                params, opt_state, _loss = update_params(
                    params, batch_image, batch_onehot, opt_state
                )

                log.add_loss(_loss)

            train_acc = compute_accuracy(params, data.train)
            test_acc = compute_accuracy(params, data.test)

            log.add_accuracy(train_acc, test_acc)

            # stopping criteria
            if jnp.abs(log.train[-1] - log.train[-2]) < tol:
                break

        ################################################################################
        # prepare output
        ################################################################################

        log.histories_asarray()

        result = NetworkResults(
            opt_state=opt_state,
            log=log,
            params=params,
        )
        return result

    return _fit


def get_optimizer(algorithm, *, step_size):

    OPTIMIZERS = {  # noqa: N806
        "adam": adam,
        "sgd": sgd,
    }

    optimizer = OPTIMIZERS[algorithm]
    optimizer = partial(optimizer, step_size=step_size)
    return optimizer


def get_update_func(get_params, opt_update, loss_func):
    @jit
    def _update_params(params, images, onehot, opt_state):
        """Compute the gradient for a batch and update the parameters"""
        value, grads = value_and_grad(loss_func)(params, images, onehot)
        opt_state = opt_update(0, grads, opt_state)
        return get_params(opt_state), opt_state, value

    return _update_params


def get_accuracy_func(forward_pass):
    def _compute_accuracy(params, data, network_output=None):
        if network_output is None:
            network_output = forward_pass(params, data.images)

        predicted = jnp.argmax(network_output, axis=1)

        _accuracy = jnp.mean(predicted == data.labels)
        return _accuracy

    return _compute_accuracy


def get_loss_func(forward_pass, loss_type):

    LOSS_FUNCTIONS = {  # noqa: N806
        "cross_entropy": _cross_entropy,
        "mean_square": _mean_square,
    }

    _loss = LOSS_FUNCTIONS[loss_type]

    @jit
    def _loss_func(params, images, onehot):
        predictions = forward_pass(params, images)
        return _loss(predictions, onehot)

    return _loss_func


def get_forward_pass_func(layer):
    @jit
    def _forward_pass(params, activations):
        # loop over hidden layers
        for w, b in params[:-1]:
            activations = layer([w, b], activations)

        # transform last layer
        last_w, last_b = params[-1]
        logits = jnp.dot(last_w, activations) + last_b
        output = softmax(logits)
        return output

    return vmap(_forward_pass, in_axes=(None, 0), out_axes=0)


def get_layer(activation_type):

    ACTIVATION_FUNCTIONS = {"sigmoid": sigmoid, "relu": relu}  # noqa: N806

    _activation_func = ACTIVATION_FUNCTIONS[activation_type]

    @jit
    def _layer(params, x):
        z = jnp.dot(params[0], x) + params[1]
        return _activation_func(z)

    return jax.jit(_layer)


def get_start_params(structure, key, scale=0.01):
    """Initialize the weights of network using Gaussian variables."""
    keys = jax.random.split(key, len(structure))

    def _initialize_layer(in_d, out_d, key):
        w_key, b_key = jax.random.split(key)
        weights = scale * jax.random.normal(w_key, (out_d, in_d))
        bias = scale * jax.random.normal(b_key, (out_d,))
        return weights, bias

    initialized = [
        _initialize_layer(in_d, out_d, _key)
        for in_d, out_d, _key in zip(structure[:-1], structure[1:], keys)
    ]
    return initialized


@jit
def _cross_entropy(predictions, onehot):
    return -jnp.sum(jnp.log(predictions) * onehot)


@jit
def _mean_square(predictions, onehot):
    return jnp.sum((predictions - onehot) ** 2)


@dataclass
class Network:
    fit: callable
    predict: callable


@dataclass
class NetworkResults:
    opt_state: OptimizerState
    log: NamedTuple
    params: List[Tuple[DeviceArray, DeviceArray]]


@dataclass
class Logging:
    # training and testing accuracy
    train: Union[List[float], DeviceArray] = field(default_factory=list)
    test: Union[List[float], DeviceArray] = field(default_factory=list)
    # training loss
    loss: Union[List[float], DeviceArray] = field(default_factory=list)

    def add_accuracy(self, train, test):
        if isinstance(self.train, list):
            self.train.append(train)
            self.test.append(test)
        else:
            self.train = jnp.append(self.train, train)
            self.test = jnp.append(self.test, test)

    def add_loss(self, loss):
        if isinstance(self.loss, list):
            self.loss.append(loss)
        else:
            self.loss = jnp.append(self.loss, loss)

    def histories_asarray(self):
        self.train = jnp.stack(self.train)
        self.test = jnp.stack(self.test)
        self.loss = jnp.stack(self.loss)
