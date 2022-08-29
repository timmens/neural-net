import pandas as pd
import pytask
from nnet.config import BLD
from nnet.simulation import get_data_simulation_kwargs
from nnet.simulation import simulation_iteration


# ======================================================================================
# Simulation specification
# ======================================================================================

SIMULATION_SPEC = {
    "linear": {
        "fitter": ("ols", "nnet"),
    },
    "linear_sparse": {
        "fitter": ("ols", "lasso", "nnet", "nnet_regularized"),
    },
    "nonlinear_sparse": {
        "fitter": ("ols", "nnet", "nnet_regularized", "boosting"),
    },
}

N_SIMULATIONS = 2  # 100

N_TEST_SAMPLES = 10_000

N_SAMPLES = [100, 1_000]  # ,10_000]

# ======================================================================================
# Simulation
# ======================================================================================

for name, specs in SIMULATION_SPEC.items():

    for n_samples in N_SAMPLES:

        for fitter in specs["fitter"]:

            _id = f"{name}-{n_samples}-{fitter}"

            task_kwargs = {
                "fitter": fitter,
                "n_samples": n_samples,
                "name": name,
                "produces": BLD.joinpath("simulation", f"{_id}.csv"),
            }

            @pytask.mark.task(id=_id, kwargs=task_kwargs)
            def task_simulation(fitter, n_samples, name, produces):

                index = ["name", "fitter", "n_samples", "iteration"]
                result = pd.DataFrame(columns=index).set_index(index)

                data_kwargs = get_data_simulation_kwargs(n_samples, name)

                for iteration in range(N_SIMULATIONS):

                    mse = simulation_iteration(
                        iteration, fitter, data_kwargs, N_TEST_SAMPLES
                    )
                    result.loc[(name, fitter, n_samples, iteration), "mse"] = mse

                result.to_csv(produces)
