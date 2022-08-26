import numpy as np
import pandas as pd
from nnet.data import simulate_data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor


# ======================================================================================
# Implemented fitting methods and simulation types
# ======================================================================================

SIMULATION_TYPES = {
    "linear": {
        "fitter": ("ols", "nnet"),
    },
    "linear_sparse": {
        "fitter": ("ridge", "lasso", "nnet", "nnet_regularized"),
    },
    "nonlinear_sparse": {
        "fitter": ("ridge", "lasso", "nnet_regularized", "boosting"),
    },
}

N_SAMPLES_GRID = [100, 500, 1_000, 10_000]

FITTER = {
    # "method": (class, kwargs)
    "ols": (LinearRegression, {"fit_intercept": False}),
    "ridge": (RidgeCV, {"fit_intercept": False}),
    "lasso": (LassoCV, {"fit_intercept": False}),
    "nnet": (
        MLPRegressor,
        {"activation": "relu", "solver": "adam", "alpha": 0.0, "max_iter": 300},
    ),
    "nnet_regularized": (
        MLPRegressor,
        {"activation": "relu", "solver": "adam", "alpha": 0.01, "max_iter": 300},
    ),
    "boosting": (GradientBoostingRegressor, {}),
}

# ======================================================================================
# Monte Carlo simulation function
# ======================================================================================


def simulation(
    n_simulations=1_000,
    simulation_types=SIMULATION_TYPES,
    n_samples_grid=N_SAMPLES_GRID,
    n_test_samples=10_000,
):

    index = ["type", "fitter", "n_samples"]
    result = pd.DataFrame(columns=index).set_index(index)

    for _type, specs in simulation_types.items():

        if _type != "linear_sparse":
            continue

        for n_samples in n_samples_grid:

            kwargs = _get_data_simulation_kwargs(n_samples, _type)

            X_test, y_test = simulate_data(
                **{**kwargs, **{"n_samples": n_test_samples}},
                seed=np.max(n_samples_grid) + n_samples,
            )

            losses = {fitter: [] for fitter in specs["fitter"]}

            for iteration in range(n_simulations):

                X_train, y_train, coef = simulate_data(
                    **kwargs, seed=iteration, return_coef=True
                )

                for fitter in specs["fitter"]:

                    model, model_kwargs = FITTER[fitter]
                    model = model(**model_kwargs).fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    _loss = mean_squared_error(y_test, y_pred)

                    losses[fitter].append(_loss)

            loss = {fitter: np.mean(_losses) for fitter, _losses in losses.items()}

            for fitter, loss in loss.items():
                result.loc[(_type, fitter, n_samples), "mean_mean_squared_error"] = loss

    result = result.sort_index()
    return result


def _get_data_simulation_kwargs(n_samples, _type):
    if _type == "linear":
        kwargs = {
            "n_samples": n_samples,
            "n_features": 30,
            "n_informative": 30,
            "noise": 0.5,
        }
    else:
        kwargs = {
            "n_samples": n_samples,
            "n_features": int(np.floor(0.1 * n_samples)),
            "n_informative": int(np.floor(0.01 * n_samples)),
            "noise": 0.5,
            "nonlinear": False,
        }

    if _type == "nonlinear":
        kwargs["nonlinear"] = True

    return kwargs
