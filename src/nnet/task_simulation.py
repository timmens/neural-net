import pytask
from nnet.config import BLD
from nnet.simulation import COMBINATIONS
from nnet.simulation import combine_task
from nnet.simulation import simulation_task


# ======================================================================================
# Run Simulation
# ======================================================================================

for comb in COMBINATIONS:

    task_kwargs = {
        "iteration": comb["iteration"],
        "fitter": comb["fitter"],
        "n_samples": comb["n_samples"],
        "name": comb["name"],
        "produces": BLD.joinpath("simulation", f"{comb['_id']}.csv"),
    }

    @pytask.mark.persist
    @pytask.mark.task(id=comb["_id"], kwargs=task_kwargs)
    def task_simulation(iteration, fitter, n_samples, name, produces):
        result = simulation_task(iteration, fitter, n_samples, name)
        result.to_csv(produces)


# ======================================================================================
# Combine Simulations
# ======================================================================================


@pytask.mark.depends_on(
    [BLD.joinpath("simulation", f"{comb['_id']}.csv") for comb in COMBINATIONS]
)
@pytask.mark.produces(BLD.joinpath("simulation", "result.csv"))
def task_combine(depends_on, produces):
    df = combine_task(depends_on.values())
    df.to_csv(produces)
