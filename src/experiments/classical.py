"""High-level experiment runners used by the notebook."""

from typing import TypedDict

import numpy as np
import numpy.typing as npt

from src.data import get_mnist_data
from src.models.classical import (
    center_data,
    closed_form,
    compute_test_error,
    compute_test_error_linear,
    compute_test_error_mod3,
    cubic_features,
    polynomial_kernel,
    principal_components,
    project_onto_PC,
    rbf_kernel,
    softmax_regression,
    update_y,
)
from src.wrappers.sklearn_svm import compute_test_error_svm, svm_predict

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]
NumberArray = npt.NDArray[np.number]


def _balanced_subsample_indices(labels: IntArray, *, n_samples: int, rng: np.random.Generator) -> IntArray:
    """
    Sample indices approximately uniformly across label values.

    Parameters
    ----------
    labels : IntArray
        Label array of shape (n,).
    n_samples : int
        Total number of indices to sample (without replacement).
    rng : np.random.Generator
        Random number generator used for sampling.

    Returns
    -------
    IntArray
        Shuffled index array of shape (n_samples,).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if labels.size == 0:
        raise ValueError("labels must be non-empty.")
    if n_samples > labels.size:
        raise ValueError("n_samples must be <= number of available samples (labels.size).")

    unique_labels = np.unique(labels)
    n_classes = unique_labels.size
    if n_samples < n_classes:
        raise ValueError("n_samples must be at least the number of classes.")

    base, remainder = divmod(n_samples, n_classes)
    per_class = np.full(n_classes, base, dtype=int)
    if remainder:
        extra = rng.choice(n_classes, size=remainder, replace=False)
        per_class[extra] += 1

    sampled_parts: list[IntArray] = []
    for idx, label in enumerate(unique_labels):
        label_indices = np.flatnonzero(labels == label)
        need = int(per_class[idx])
        if label_indices.size < need:
            raise ValueError(f"Not enough examples for label={label}: need {need}, have {label_indices.size}.")
        sampled_parts.append(rng.choice(label_indices, size=need, replace=False))

    sampled = np.concatenate(sampled_parts)
    rng.shuffle(sampled)
    return sampled


class KernelSoftmaxInfo(TypedDict):
    """Artifacts returned by the kernelised softmax experiment runner."""

    pcs: FloatArray
    feature_means: FloatArray
    train_indices: IntArray


def run_linear_regression_on_mnist(*, lambda_factor: float = 1.0) -> float:
    """
    Train ridge regression baseline and return test error.

    Parameters
    ----------
    lambda_factor : float, default=1.0
        L2 regularisation strength.

    Returns
    -------
    float
        Fraction of misclassified examples on the test set.
    """
    train_x, train_y, test_x, test_y = get_mnist_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x_bias, train_y, lambda_factor)
    return float(compute_test_error_linear(test_x_bias, test_y, theta))


def run_svm_one_vs_rest_on_mnist(*, C: float = 0.1, random_state: int = 0) -> float:
    """
    Binary linear SVM (one-vs-rest: digit 0 vs not-0) test error.

    Parameters
    ----------
    C : float, default=0.1
        SVM regularisation parameter.
    random_state : int, default=0
        Random seed for sklearn.

    Returns
    -------
    float
        Fraction of misclassified examples on the test set.
    """
    train_x, train_y, test_x, test_y = get_mnist_data()
    train_y = train_y.copy()
    test_y = test_y.copy()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = svm_predict(train_x, train_y, test_x, random_state=random_state, C=C, kernel="linear")
    return float(compute_test_error_svm(test_y, pred_test_y))


def run_multiclass_svm_on_mnist(*, C: float = 0.1, random_state: int = 0) -> float:
    """
    Multiclass linear SVM (one-vs-rest) test error.

    Parameters
    ----------
    C : float, default=0.1
        SVM regularisation parameter.
    random_state : int, default=0
        Random seed for sklearn.

    Returns
    -------
    float
        Fraction of misclassified examples on the test set.

    Notes
    -----
    This uses ``LinearSVC``, which implements multiclass classification via a one-vs-rest strategy.
    """
    train_x, train_y, test_x, test_y = get_mnist_data()
    pred_test_y = svm_predict(train_x, train_y, test_x, random_state=random_state, C=C, kernel="linear")
    return float(compute_test_error_svm(test_y, pred_test_y))


def run_softmax_on_mnist(
    *,
    temp_parameter: float = 1.0,
    alpha: float = 0.3,
    lambda_factor: float = 1.0e-4,
    num_iterations: int = 150,
    save_theta_path: str | None = None,
) -> tuple[FloatArray, list[float], float]:
    """
    Train softmax regression on raw pixels.

    Parameters
    ----------
    temp_parameter : float, default=1.0
        Temperature parameter tau.
    alpha : float, default=0.3
        Learning rate.
    lambda_factor : float, default=1.0e-4
        L2 regularisation strength.
    num_iterations : int, default=150
        Gradient descent iterations.
    save_theta_path : str | None, default=None
        Destination path to save the learned weights (recommended extension: ``.npz``).

    Returns
    -------
    tuple[FloatArray, list[float], float]
        Learned parameters, cost history, and test error.
    """
    train_x, train_y, test_x, test_y = get_mnist_data()
    theta, cost_history = softmax_regression(
        train_x,
        train_y,
        temp_parameter=temp_parameter,
        alpha=alpha,
        lambda_factor=lambda_factor,
        k=10,
        num_iterations=num_iterations,
    )
    test_error = float(compute_test_error(test_x, test_y, theta, temp_parameter))

    if save_theta_path is not None:
        np.savez_compressed(save_theta_path, weights=theta)

    return theta, cost_history, test_error


def run_softmax_on_mnist_mod3(
    *,
    temp_parameter: float = 1.0,
    alpha: float = 0.3,
    lambda_factor: float = 1.0e-4,
    num_iterations: int = 150,
    save_theta_path: str | None = None,
) -> tuple[FloatArray, list[float], float]:
    """
    Train softmax regression on labels mod 3.

    Parameters
    ----------
    temp_parameter : float, default=1.0
        Temperature parameter tau.
    alpha : float, default=0.3
        Learning rate.
    lambda_factor : float, default=1.0e-4
        L2 regularisation strength.
    num_iterations : int, default=150
        Gradient descent iterations.
    save_theta_path : str | None, default=None
        Destination path to save the learned weights (recommended extension: ``.npz``).

    Returns
    -------
    tuple[FloatArray, list[float], float]
        Learned parameters, cost history, and test error.
    """
    train_x, train_y, test_x, test_y = get_mnist_data()
    train_y_mod3, test_y_mod3 = update_y(train_y, test_y)
    theta, cost_history = softmax_regression(
        train_x,
        train_y_mod3,
        temp_parameter=temp_parameter,
        alpha=alpha,
        lambda_factor=lambda_factor,
        k=3,
        num_iterations=num_iterations,
    )
    test_error = float(compute_test_error(test_x, test_y_mod3, theta, temp_parameter))
    if save_theta_path is not None:
        np.savez_compressed(save_theta_path, weights=theta)
    return theta, cost_history, test_error


def run_softmax_on_mnist_eval_mod3(
    *,
    temp_parameter: float = 1.0,
    alpha: float = 0.3,
    lambda_factor: float = 1.0e-4,
    num_iterations: int = 150,
    save_theta_path: str | None = None,
) -> tuple[FloatArray, list[float], float]:
    """
    Train softmax regression on digits, evaluate test error on labels mod 3.

    Parameters
    ----------
    temp_parameter : float, default=1.0
        Temperature parameter tau.
    alpha : float, default=0.3
        Learning rate.
    lambda_factor : float, default=1.0e-4
        L2 regularisation strength.
    num_iterations : int, default=150
        Gradient descent iterations.
    save_theta_path : str | None, default=None
        Destination path to save the learned weights (recommended extension: ``.npz``).

    Returns
    -------
    tuple[FloatArray, list[float], float]
        Learned parameters, cost history, and mod-3 test error.
    """
    train_x, train_y, test_x, test_y = get_mnist_data()
    theta, cost_history = softmax_regression(
        train_x,
        train_y,
        temp_parameter=temp_parameter,
        alpha=alpha,
        lambda_factor=lambda_factor,
        k=10,
        num_iterations=num_iterations,
    )

    _, test_y_mod3 = update_y(train_y, test_y)
    test_error_mod3 = float(compute_test_error_mod3(test_x, test_y_mod3, theta, temp_parameter))

    if save_theta_path is not None:
        np.savez_compressed(save_theta_path, weights=theta)

    return theta, cost_history, test_error_mod3


def compute_pca_representations(
    *,
    n_components: int,
) -> tuple[FloatArray, IntArray, FloatArray, IntArray, FloatArray, FloatArray]:
    """
    Compute PCA representations for train and test sets.

    Parameters
    ----------
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    tuple[FloatArray, IntArray, FloatArray, IntArray, FloatArray, FloatArray]
        PCA representations (train_pca, train_y, test_pca, test_y) and PCA artifacts (pcs, feature_means).
    """
    train_x, train_y, test_x, test_y = get_mnist_data()
    train_x_centered, feature_means = center_data(train_x)
    pcs = principal_components(train_x_centered)
    train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)
    return train_pca, train_y, test_pca, test_y, pcs, feature_means


def run_softmax_on_mnist_pca(
    *,
    n_components: int = 18,
    temp_parameter: float = 1.0,
    alpha: float = 0.3,
    lambda_factor: float = 1.0e-4,
    num_iterations: int = 150,
) -> tuple[FloatArray, list[float], float, dict[str, FloatArray]]:
    """
    Train softmax regression on PCA-reduced features.

    Parameters
    ----------
    n_components : int, default=18
        Number of PCA components.
    temp_parameter : float, default=1.0
        Temperature parameter tau.
    alpha : float, default=0.3
        Learning rate.
    lambda_factor : float, default=1.0e-4
        L2 regularisation strength.
    num_iterations : int, default=150
        Gradient descent iterations.

    Returns
    -------
    tuple[FloatArray, list[float], float, dict[str, FloatArray]]
        Learned parameters, cost history, test error, and PCA info dict.
    """
    train_pca, train_y, test_pca, test_y, pcs, feature_means = compute_pca_representations(n_components=n_components)
    theta, cost_history = softmax_regression(
        train_pca,
        train_y,
        temp_parameter=temp_parameter,
        alpha=alpha,
        lambda_factor=lambda_factor,
        k=10,
        num_iterations=num_iterations,
    )
    test_error = float(compute_test_error(test_pca, test_y, theta, temp_parameter))
    pca_info = {"pcs": pcs, "feature_means": feature_means}
    return theta, cost_history, test_error, pca_info


def run_softmax_on_mnist_cubic(
    *,
    temp_parameter: float = 1.0,
    alpha: float = 0.3,
    lambda_factor: float = 1.0e-4,
    num_iterations: int = 150,
) -> tuple[FloatArray, list[float], float, dict[str, FloatArray]]:
    """
    Train softmax regression on cubic features of 10D PCA representation.

    Parameters
    ----------
    temp_parameter : float, default=1.0
        Temperature parameter tau.
    alpha : float, default=0.3
        Learning rate.
    lambda_factor : float, default=1.0e-4
        L2 regularisation strength.
    num_iterations : int, default=150
        Gradient descent iterations.

    Returns
    -------
    tuple[FloatArray, list[float], float, dict[str, FloatArray]]
        Learned parameters, cost history, test error, and info dict with PCA artifacts.
    """
    train_pca10, train_y, test_pca10, test_y, pcs, feature_means = compute_pca_representations(n_components=10)
    train_cube = cubic_features(train_pca10)
    test_cube = cubic_features(test_pca10)
    theta, cost_history = softmax_regression(
        train_cube,
        train_y,
        temp_parameter=temp_parameter,
        alpha=alpha,
        lambda_factor=lambda_factor,
        k=10,
        num_iterations=num_iterations,
    )
    test_error = float(compute_test_error(test_cube, test_y, theta, temp_parameter))
    info = {"pcs": pcs, "feature_means": feature_means, "train_pca10": train_pca10, "test_pca10": test_pca10}
    return theta, cost_history, test_error, info


def run_sklearn_svm_poly_on_pca10(*, random_state: int = 0, degree: int = 3) -> float:
    """
    Train sklearn SVC with polynomial kernel on 10D PCA features.

    Parameters
    ----------
    random_state : int, default=0
        Random seed for sklearn.
    degree : int, default=3
        Polynomial kernel degree.

    Returns
    -------
    float
        Fraction of misclassified test examples.
    """
    train_pca10, train_y, test_pca10, test_y, _pcs, _feature_means = compute_pca_representations(n_components=10)
    pred = svm_predict(
        train_pca10,
        train_y,
        test_pca10,
        random_state=random_state,
        C=1.0,
        kernel="poly",
        degree=degree,
    )
    return compute_test_error_svm(test_y, pred)


def run_sklearn_svm_rbf_on_pca10(*, random_state: int = 0) -> float:
    """
    Train sklearn SVC with RBF kernel on 10D PCA features.

    Parameters
    ----------
    random_state : int, default=0
        Random seed for sklearn.

    Returns
    -------
    float
        Fraction of misclassified test examples.
    """
    train_pca10, train_y, test_pca10, test_y, _pcs, _feature_means = compute_pca_representations(n_components=10)
    pred = svm_predict(
        train_pca10,
        train_y,
        test_pca10,
        random_state=random_state,
        C=1.0,
        kernel="rbf",
    )
    return compute_test_error_svm(test_y, pred)


def run_kernelised_softmax_on_mnist_pca10(
    *,
    subset_size: int = 1000,
    kernel: str = "rbf",
    gamma: float = 0.5,
    c: float = 1.0,
    p: int = 2,
    temp_parameter: float = 1.0,
    alpha: float = 0.3,
    lambda_factor: float = 1.0e-4,
    num_iterations: int = 50,
    pbar: bool = False,
    random_state: int = 0,
) -> tuple[FloatArray, list[float], float, KernelSoftmaxInfo]:
    """
    Train kernelised softmax regression on PCA-10 MNIST features using a balanced subset.

    Parameters
    ----------
    subset_size : int, default=1000
        Number of training examples to use when forming the Gram matrix.
    kernel : str, default="rbf"
        Which kernel to use: "rbf" or "poly".
    gamma : float, default=0.5
        RBF kernel bandwidth parameter.
    c : float, default=1.0
        Polynomial kernel coefficient.
    p : int, default=2
        Polynomial kernel degree.
    temp_parameter : float, default=1.0
        Temperature parameter tau.
    alpha : float, default=0.3
        Learning rate.
    lambda_factor : float, default=1.0e-4
        L2 regularisation strength.
    num_iterations : int, default=50
        Gradient descent iterations.
    pbar : bool, default=False
        Whether to show a progress bar during training.
    random_state : int, default=0
        Random seed controlling the balanced subsample.

    Returns
    -------
    tuple[FloatArray, list[float], float, KernelSoftmaxInfo]
        Learned parameters, cost history, test error, and PCA/subsample artifacts.

    Notes
    -----
    This experiment uses the Gram matrix over a subset of PCA-10 training points as the
    feature representation. This scales quadratically in the subset size, so it is
    intended as a demonstration rather than a full-scale MNIST baseline.
    """
    train_pca10, train_y, test_pca10, test_y, pcs, feature_means = compute_pca_representations(n_components=10)

    rng = np.random.default_rng(random_state)
    train_indices = _balanced_subsample_indices(train_y, n_samples=subset_size, rng=rng)
    sub_train_x = train_pca10[train_indices]
    sub_train_y = train_y[train_indices]

    kernel_lower = kernel.strip().lower()
    if kernel_lower == "rbf":
        K_train = rbf_kernel(sub_train_x, sub_train_x, gamma=gamma)
        K_test = rbf_kernel(test_pca10, sub_train_x, gamma=gamma)
    elif kernel_lower in {"poly", "polynomial"}:
        K_train = polynomial_kernel(sub_train_x, sub_train_x, c=c, p=p)
        K_test = polynomial_kernel(test_pca10, sub_train_x, c=c, p=p)
    else:
        raise ValueError(f"Unsupported kernel: {kernel!r}. Use 'rbf' or 'poly'.")

    theta, cost_history = softmax_regression(
        K_train,
        sub_train_y,
        temp_parameter=temp_parameter,
        alpha=alpha,
        lambda_factor=lambda_factor,
        k=10,
        num_iterations=num_iterations,
        pbar=pbar,
    )
    test_error = float(compute_test_error(K_test, test_y, theta, temp_parameter))
    info: KernelSoftmaxInfo = {"pcs": pcs, "feature_means": feature_means, "train_indices": train_indices}
    return theta, cost_history, test_error, info
