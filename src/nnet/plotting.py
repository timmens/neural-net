"""Plotting code.

The functions are inspired and copied from
https://github.com/RobertTLange/code-and-blog/tree/master/04_jax_intro

"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(
    context="poster",
    style="white",
    font="sans-serif",
    font_scale=1,
    color_codes=True,
    rc=None,
)


def plot_examples(data, *, n_examples=4, figsize=(10, 5), fontsize=30, random=True):

    if random:
        _ids = np.random.choice(len(data.labels), size=n_examples, replace=False)
    else:
        _ids = np.arange(n_examples)

    images = data.images[_ids].reshape(4, 28, 28)
    labels = data.labels[_ids]

    if data.predictions is not None:
        predictions = data.predictions[_ids]
    else:
        predictions = None

    fig, axs = plt.subplots(1, n_examples, figsize=figsize)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i], cmap="Greys")
        if predictions is None:
            title = f"Label: {labels[i]}"
        else:
            title = f"Label: {labels[i]}\nPred: {predictions[i]}"
        ax.set_title(title, fontsize=fontsize)
        ax.set_axis_off()

    fig.tight_layout()
    return None


def plot_performance(log):

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(log.loss)
    axs[0].set_xlabel("# Batch Updates")
    axs[0].set_ylabel("Batch Loss")
    axs[0].set_title("Training Loss")

    axs[1].plot(log.train, label="Training")
    axs[1].plot(log.test, label="Test")
    axs[1].set_xlabel("# Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Prediction Accuracy")
    axs[1].legend()

    for i in range(2):
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)

    fig.tight_layout()
    return None
