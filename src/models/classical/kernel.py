"""Kernel functions."""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]


def polynomial_kernel(X: FloatArray, Y: FloatArray, c: float, p: int) -> FloatArray:
    """
    Compute the polynomial kernel matrix.

    Parameters
    ----------
    X : FloatArray
        Feature matrix of shape (n, d).
    Y : FloatArray
        Feature matrix of shape (m, d).
    c : float
        Polynomial kernel coefficient.
    p : int
        Polynomial degree.

    Returns
    -------
    FloatArray
        Kernel matrix of shape (n, m) with entries (x·y + c)^p.
    """
    return (X @ Y.T + c) ** p


def rbf_kernel(X: FloatArray, Y: FloatArray, gamma: float) -> FloatArray:
    """
    Compute the Gaussian RBF kernel matrix.

    Parameters
    ----------
    X : FloatArray
        Feature matrix of shape (n, d).
    Y : FloatArray
        Feature matrix of shape (m, d).
    gamma : float
        RBF bandwidth parameter.

    Returns
    -------
    FloatArray
        Kernel matrix of shape (n, m) with entries exp(-gamma ||x - y||^2).
    """
    X_norm_squared = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_norm_squared = np.sum(Y**2, axis=1).reshape(1, -1)
    dist_squared = X_norm_squared + Y_norm_squared - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * dist_squared)
