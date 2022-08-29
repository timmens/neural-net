import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from nnet.data import simulate_data
from nnet.network import CustomMLP
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor


# ======================================================================================
# Simulation specification
# ======================================================================================

SIMULATION_SPEC = {
    "linear": {
        "fitter": ("ols", "nnet", "boosting"),
    },
    "linear_sparse": {
        "fitter": ("ols", "lasso", "nnet", "nnet_regularized"),
    },
    "nonlinear_sparse": {
        "fitter": ("ols", "nnet", "nnet_regularized", "boosting"),
    },
}

N_SIMULATIONS = 50

N_TEST_SAMPLES = 10_000

N_SAMPLES_GRID = [100, 1_000, 10_000]

ALPHAS = [0.001, 0.01, 0.1]

# ======================================================================================
# Create Simulation Combinations
# ======================================================================================

COMBINATIONS = []
for name, specs in SIMULATION_SPEC.items():
    for n_samples in N_SAMPLES_GRID:
        for fitter in specs["fitter"]:
            for iteration in range(N_SIMULATIONS):
                _id = f"{name}-{n_samples}-{fitter}-{iteration}"
                comb = {
                    "name": name,
                    "n_samples": n_samples,
                    "fitter": fitter,
                    "iteration": iteration,
                    "_id": _id,
                }
                COMBINATIONS.append(comb)


def get_data_kwargs(n_samples, _type):
    if "sparse" in _type:
        kwargs = {
            "n_samples": n_samples,
            "n_features": int(np.floor(0.1 * n_samples)),
            "n_informative": int(np.floor(0.01 * n_samples)),
            "noise": 0.5,
        }
        if "nonlinear" in _type:
            kwargs["nonlinear"] = True
    else:
        kwargs = {
            "n_samples": n_samples,
            "n_features": 30,
            "n_informative": 30,
            "noise": 0.5,
            "nonlinear": False,
        }
    return kwargs


FITTER = {
    # "method": (class, kwargs)
    "ols": (LinearRegression, {"fit_intercept": False, "copy_X": False}),
    "ridge": (RidgeCV, {"fit_intercept": False, "cv": 3, "alphas": ALPHAS}),
    "lasso": (
        LassoCV,
        {"fit_intercept": False, "cv": 3, "n_jobs": -1, "alphas": ALPHAS},
    ),
    "nnet": (CustomMLP, {"n_epochs": 100, "sparsity_level": 0.01}),
    "nnet_regularized": (
        CustomMLP,
        {"n_epochs": 100, "sparsity_level": 0.01, "l1_penalty": 0.05},
    ),
    "nnet_sklearn": (
        MLPRegressor,
        {"activation": "relu", "solver": "adam", "alpha": 0.0, "max_iter": 300},
    ),
    "nnet_sklearn_regularized": (
        MLPRegressor,
        {"activation": "relu", "solver": "adam", "alpha": 0.01, "max_iter": 300},
    ),
    "boosting": (
        CatBoostRegressor,
        {"iterations": 1_000, "depth": 2, "rsm": 0.2, "silent": True},
    ),
}

# ======================================================================================
# Simulation function
# ======================================================================================


INDEX = ["name", "fitter", "n_samples", "iteration"]


def simulation_task(iteration, fitter, n_samples, name):
    index = ["name", "fitter", "n_samples", "iteration"]
    result = pd.DataFrame(columns=index).set_index(index)

    data_kwargs = get_data_kwargs(n_samples, name)

    mse = simulation_iteration(iteration, fitter, data_kwargs, N_TEST_SAMPLES)
    result.loc[(name, fitter, n_samples, iteration), "mse"] = mse

    return result


def simulation(
    n_simulations=1_000,
    *,
    simulation_types,
    n_samples_grid,
    n_test_samples,
    aggregation_methods,
):

    index = ["type", "fitter", "n_samples"]
    result = pd.DataFrame(columns=index).set_index(index)

    for _type, specs in simulation_types.items():

        for n_samples in n_samples_grid:

            # ==========================================================================
            # Simulate testing data (used to approximate integral)

            data_kwargs = get_data_kwargs(n_samples, _type)

            X_test, y_test = simulate_data(
                **{**data_kwargs, **{"n_samples": n_test_samples}},
                seed=np.max(n_samples_grid) + n_samples,
            )

            # ==========================================================================
            # Loop over fitting methods and Monte Carlo iterations

            for fitter in specs["fitter"]:

                loss = []

                for iteration in range(n_simulations):

                    _loss = simulation_iteration(
                        iteration, fitter, data_kwargs, X_test, y_test
                    )
                    loss.append(_loss)

                # Compute aggregation metrics
                for name, method in aggregation_methods.items():
                    result.loc[(_type, fitter, n_samples), f"{name}_mse"] = method(loss)

    result = result.sort_index()
    return result


def simulation_iteration(iteration, fitter, data_kwargs, n_test_samples):
    # Simulate testing data
    test_data_kwargs = {**data_kwargs, **{"n_samples": n_test_samples}}
    X_test, y_test = simulate_data(
        **test_data_kwargs,
        seed=20_000,
    )

    # Simulate training data
    X_train, y_train = simulate_data(**data_kwargs, seed=iteration)

    # Get fitting method (model) and train
    model, model_kwargs = FITTER[fitter]
    model = model(**model_kwargs).fit(X_train, y_train)

    # Predict on testing data
    y_pred = model.predict(X_test)

    # Compute metrics (here: mean squared error)
    loss = mean_squared_error(y_test, y_pred)
    return loss


# ======================================================================================
# Combine Simulation Results
# ======================================================================================


def combine_task(paths):
    dfs = [pd.read_csv(path, index_col=INDEX) for path in paths]
    df = pd.concat(dfs).sort_index()
    return df
