"""Temperature-parameter analysis for softmax regression."""

import numpy as np
import numpy.typing as npt

from src.data import get_mnist_data
from src.models.classical import compute_probabilities, softmax_regression

FloatArray = npt.NDArray[np.floating]


def analyse_temperature_effects(
    temp_values: FloatArray,
    *,
    sample_size: int = 1000,
    num_iterations: int = 100,
    alpha: float = 0.3,
    lambda_factor: float = 1.0e-4,
    k: int = 10,
    random_state: int = 0,
) -> dict[float, dict[str, float]]:
    """
    Analyse how temperature affects parameter magnitude and prediction entropy.

    Trains softmax regression at each temperature value and measures:
    - Frobenius norm of the weight matrix (parameter magnitude)
    - Average prediction entropy across test examples

    Parameters
    ----------
    temp_values : FloatArray
        Array of temperature values to evaluate.
    sample_size : int, default=1000
        Number of training examples to sample for each run.
    num_iterations : int, default=100
        Gradient descent iterations for each temperature.
    alpha : float, default=0.3
        Learning rate for gradient descent.
    lambda_factor : float, default=1.0e-4
        L2 regularisation strength.
    k : int, default=10
        Number of output classes.
    random_state : int, default=0
        Random seed for reproducible sampling.

    Returns
    -------
    dict[float, dict[str, float]]
        Mapping from temperature to metrics dict containing:
        - ``"theta_magnitude"``: Frobenius norm of learned weights
        - ``"avg_entropy"``: Mean prediction entropy on test set
    """
    rng = np.random.default_rng(random_state)
    train_x, train_y, test_x, _test_y = get_mnist_data()

    results: dict[float, dict[str, float]] = {}
    sample_indices = rng.choice(train_x.shape[0], sample_size, replace=False)
    train_x_sample = train_x[sample_indices]
    train_y_sample = train_y[sample_indices]

    for temp in temp_values:
        theta, _ = softmax_regression(
            train_x_sample,
            train_y_sample,
            temp_parameter=float(temp),
            alpha=alpha,
            lambda_factor=lambda_factor,
            k=k,
            num_iterations=num_iterations,
        )

        theta_magnitude = np.linalg.norm(theta)
        X_augmented = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
        probs = compute_probabilities(X_augmented, theta, float(temp))

        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=0)
        avg_entropy = float(np.mean(entropy))

        results[float(temp)] = {"theta_magnitude": float(theta_magnitude), "avg_entropy": avg_entropy}

    return results
