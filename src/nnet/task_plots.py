import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns
from nnet.config import BLD


@pytask.mark.depends_on(BLD.joinpath("simulation", "result.csv"))
@pytask.mark.produces(BLD.joinpath("simulation_result.png"))
def task_paper_plot(depends_on, produces):

    df = pd.read_csv(depends_on)
    df["mse"] -= 0.25

    RENAME = {
        "ols": "OLS",
        "lasso": "Lasso",
        "nnet": "NNet",
        "nnet_regularized": "RegNNet",
        "boosting": "Boosting",
    }

    PALETTE = {
        "OLS": "tab:blue",
        "Lasso": "tab:orange",
        "NNet": "tab:green",
        "RegNNet": "tab:red",
        "Boosting": "tab:gray",
    }

    def kde_plot(df, name, n_samples, ax, palette=PALETTE, rename=RENAME):
        df = df.query("name == @name & n_samples == @n_samples")
        df = df.replace(to_replace=rename)
        ax = sns.kdeplot(
            x="mse",
            data=df,
            hue="fitter",
            fill=True,
            legend=True,
            palette=palette,
            ax=ax,
        )
        ax.set_xlabel("Adjusted MSE")
        ax.yaxis.set_ticklabels([])
        legend = ax.get_legend()
        legend.set_title(None)
        legend.set_frame_on(False)
        sns.despine(ax=ax, offset=5, trim=False)
        font = {"family": "serif", "size": 8}
        matplotlib.rc("font", **font)
        return None

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i, name in enumerate(["linear", "linear_sparse", "nonlinear_sparse"]):
        for j, n_samples in enumerate([100, 1_000, 10_000]):

            kde_plot(df, name, n_samples, ax=axes[i, j])

    fig.savefig(produces, transparent=True, bbox_inches="tight", pad_inches=0)
