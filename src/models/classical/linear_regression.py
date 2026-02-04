"""Linear regression baseline."""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]
NumberArray = npt.NDArray[np.number]


def closed_form(X: FloatArray, Y: NumberArray, lambda_factor: float) -> FloatArray:
    """
    Compute the closed-form ridge regression solution.

    Parameters
    ----------
    X : FloatArray
        Feature matrix with bias column included, shape (n, d + 1).
    Y : NumberArray
        Target labels, shape (n,).
    lambda_factor : float
        L2 regularisation strength.

    Returns
    -------
    FloatArray
        Regression weights including bias term, shape (d + 1,).
    """
    d = X.shape[1] - 1
    left = X.T @ X + lambda_factor * np.eye(d + 1)
    right = X.T @ Y
    theta = np.linalg.solve(left, right).astype(float)
    return theta


def compute_test_error_linear(test_x: FloatArray, Y: NumberArray, theta: FloatArray) -> float:
    """
    Compute classification-style error after rounding predictions to [0, 9].

    Parameters
    ----------
    test_x : FloatArray
        Test feature matrix with bias column, shape (m, d + 1).
    Y : NumberArray
        Ground-truth digit labels, shape (m,).
    theta : FloatArray
        Regression weights including bias term, shape (d + 1,).

    Returns
    -------
    float
        Fraction of misclassified examples after rounding.
    """
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict = np.clip(test_y_predict, 0, 9)
    return float(1 - np.mean(test_y_predict == Y))
