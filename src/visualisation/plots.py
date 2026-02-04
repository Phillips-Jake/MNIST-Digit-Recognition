"""Plotting helpers used by the notebook and experiment runners."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from src.models.classical.features import project_onto_PC

FloatArray = npt.NDArray[np.floating]
NumberArray = npt.NDArray[np.number]


def plot_PC(
    X: FloatArray,
    pcs: FloatArray,
    labels: NumberArray,
    feature_means: FloatArray,
) -> None:
    """
    Scatter plot of samples projected onto the first two principal components.

    Parameters
    ----------
    X : FloatArray
        Input data of shape (n, d).
    pcs : FloatArray
        Principal component matrix of shape (d, d).
    labels : NumberArray
        Labels of shape (n,).
    feature_means : FloatArray
        Feature means of shape (d,).
    """
    pc_data = project_onto_PC(X, pcs, n_components=2, feature_means=feature_means)
    text_labels = [str(z) for z in labels.tolist()]
    fig, ax = plt.subplots()
    ax.scatter(pc_data[:, 0], pc_data[:, 1], alpha=0, marker=".")
    for i, txt in enumerate(text_labels):
        ax.annotate(txt, (pc_data[i, 0], pc_data[i, 1]))
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.show()
