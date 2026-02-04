"""Multinomial (softmax) regression implemented with batch gradient descent."""

import numpy as np
import numpy.typing as npt
from tqdm import trange

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]
NumberArray = npt.NDArray[np.number]


def augment_feature_vector(X: FloatArray) -> FloatArray:
    """
    Prepend a bias feature (a column of ones) to each data point.

    Parameters
    ----------
    X : FloatArray
        Input data of shape (n, d).

    Returns
    -------
    FloatArray
        Bias-augmented data of shape (n, d + 1).
    """
    column_of_ones = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack((column_of_ones, X))


def compute_probabilities(X: FloatArray, theta: FloatArray, temp_parameter: float) -> FloatArray:
    """
    Compute softmax probabilities for each example.

    Parameters
    ----------
    X : FloatArray
        Bias-augmented data matrix of shape (n, d + 1).
    theta : FloatArray
        Model parameters of shape (k, d + 1).
    temp_parameter : float
        Temperature parameter tau.

    Returns
    -------
    FloatArray
        Probability matrix of shape (k, n), where entry (j, i) is P(y=j | x^(i)).
    """
    logits = (theta @ X.T) / temp_parameter
    logits -= np.max(logits, axis=0, keepdims=True)
    np.exp(logits, out=logits)
    logits /= np.sum(logits, axis=0, keepdims=True)
    return logits


def compute_cost_function(
    X: FloatArray,
    Y: IntArray,
    theta: FloatArray,
    lambda_factor: float,
    temp_parameter: float,
) -> float:
    """
    Compute the regularised negative log-likelihood objective.

    Parameters
    ----------
    X : FloatArray
        Bias-augmented data matrix of shape (n, d + 1).
    Y : IntArray
        Labels of shape (n,), integer encoded in [0, k-1].
    theta : FloatArray
        Model parameters of shape (k, d + 1).
    lambda_factor : float
        L2 regularisation strength.
    temp_parameter : float
        Temperature parameter tau.

    Returns
    -------
    float
        Scalar objective value.
    """
    n = X.shape[0]
    Z = compute_probabilities(X, theta, temp_parameter)
    # Implicit OHE: index the true class per example instead of building a one-hot matrix.
    correct_class_probabilities = Z[Y, np.arange(n)]
    cost = -np.sum(np.log(correct_class_probabilities)) / n
    regularisation = (lambda_factor / 2) * np.sum(theta**2)
    return float(cost + regularisation)


def run_gradient_descent_iteration(
    X: FloatArray,
    Y: IntArray,
    theta: FloatArray,
    alpha: float,
    lambda_factor: float,
    temp_parameter: float,
) -> FloatArray:
    """
    Run one batch gradient descent update step.

    Parameters
    ----------
    X : FloatArray
        Bias-augmented data matrix of shape (n, d + 1).
    Y : IntArray
        Labels of shape (n,), integer encoded in [0, k-1].
    theta : FloatArray
        Model parameters of shape (k, d + 1).
    alpha : float
        Learning rate.
    lambda_factor : float
        L2 regularisation strength.
    temp_parameter : float
        Temperature parameter tau.

    Returns
    -------
    FloatArray
        Updated parameters of shape (k, d + 1).
    """
    n = X.shape[0]
    Z = compute_probabilities(X, theta, temp_parameter)
    # In-place equivalent of subtracting a one-hot label matrix.
    Z[Y, np.arange(n)] -= 1
    grad_J = (1 / (temp_parameter * n)) * (Z @ X) + lambda_factor * theta
    return theta - alpha * grad_J


def update_y(train_y: IntArray, test_y: IntArray) -> tuple[IntArray, IntArray]:
    """
    Convert digit labels to (mod 3) labels for train and test arrays.

    Parameters
    ----------
    train_y : IntArray
        Training labels of shape (n,).
    test_y : IntArray
        Test labels of shape (m,).

    Returns
    -------
    tuple[IntArray, IntArray]
        Labels reduced modulo 3.
    """
    return train_y % 3, test_y % 3


def compute_test_error_mod3(
    X: FloatArray,
    Y: IntArray,
    theta: FloatArray,
    temp_parameter: float,
) -> float:
    """
    Compute error rate for mod-3 labels using a digit classifier's predictions.

    Parameters
    ----------
    X : FloatArray
        Input data of shape (n, d).
    Y : IntArray
        Mod-3 labels of shape (n,).
    theta : FloatArray
        Model parameters of shape (k, d + 1).
    temp_parameter : float
        Temperature parameter tau.

    Returns
    -------
    float
        Fraction of misclassified examples under mod-3 evaluation.
    """
    assigned_labels = get_classification(X, theta, temp_parameter)
    return float(1 - np.mean((assigned_labels % 3) == (Y % 3)))


def softmax_regression(
    X: FloatArray,
    Y: IntArray,
    temp_parameter: float,
    alpha: float,
    lambda_factor: float,
    k: int,
    num_iterations: int,
    *,
    pbar: bool = False,
) -> tuple[FloatArray, list[float]]:
    """
    Train softmax regression via batch gradient descent with theta initialised to zeros.

    Parameters
    ----------
    X : FloatArray
        Input data of shape (n, d).
    Y : IntArray
        Labels of shape (n,), integer encoded in [0, k-1].
    temp_parameter : float
        Temperature parameter tau.
    alpha : float
        Learning rate.
    lambda_factor : float
        L2 regularisation strength.
    k : int
        Number of classes.
    num_iterations : int
        Number of gradient descent iterations.
    pbar : bool, default=False
        Whether to show a progress bar during training.

    Returns
    -------
    tuple[FloatArray, list[float]]
        Learned parameters of shape (k, d + 1) and cost value after each iteration.
    """
    X = augment_feature_vector(X)
    theta: FloatArray = np.zeros([k, X.shape[1]], dtype=float)
    cost_function_progression: list[float] = []
    for _ in range(num_iterations) if not pbar else trange(num_iterations, desc="Training Softmax"):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression


def get_classification(X: FloatArray, theta: FloatArray, temp_parameter: float) -> IntArray:
    """
    Predict labels for each example in X.

    Parameters
    ----------
    X : FloatArray
        Input data of shape (n, d).
    theta : FloatArray
        Model parameters of shape (k, d + 1).
    temp_parameter : float
        Temperature parameter tau.

    Returns
    -------
    IntArray
        Predicted labels of shape (n,), integer encoded in [0, k-1].
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis=0)


def compute_test_error(
    X: FloatArray,
    Y: IntArray,
    theta: FloatArray,
    temp_parameter: float,
) -> float:
    """
    Compute the fraction of misclassified examples.

    Parameters
    ----------
    X : FloatArray
        Input data of shape (n, d).
    Y : IntArray
        Labels of shape (n,).
    theta : FloatArray
        Model parameters of shape (k, d + 1).
    temp_parameter : float
        Temperature parameter tau.

    Returns
    -------
    float
        Fraction of misclassified examples.
    """
    assigned_labels = get_classification(X, theta, temp_parameter)
    return float(1 - np.mean(assigned_labels == Y))
