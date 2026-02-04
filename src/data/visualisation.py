"""Matplotlib-based visualisations for MNIST examples."""

import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

NumberArray = npt.NDArray[np.number]


def plot_images(X: NumberArray) -> None:
    """
    Plot one or more flattened 28x28 MNIST images.

    Parameters
    ----------
    X : NumberArray
        Array of images with shape (n, 784) or a single image with shape (784,).
    """
    if X.ndim == 1:
        X = np.array([X])
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one image.")
    if X.shape[1] != 28 * 28:
        raise ValueError(f"Expected flattened images of length 784, got shape {X.shape!r}.")
    num_images = X.shape[0]
    num_rows = math.floor(math.sqrt(num_images))
    num_cols = math.ceil(num_images / num_rows)
    cmap = cm.get_cmap("Greys_r")
    for i in range(num_images):
        reshaped_image = X[i, :].reshape(28, 28)
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(reshaped_image, cmap=cmap)
        plt.axis("off")
    plt.show()
