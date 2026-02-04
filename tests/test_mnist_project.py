"""Pytest suite for the MNIST project modules."""

import numpy as np
import numpy.testing as npt

from src.data import get_mnist_data
from src.models.classical.features import center_data, cubic_features, principal_components, project_onto_PC
from src.models.classical.kernel import polynomial_kernel, rbf_kernel
from src.models.classical.linear_regression import closed_form
from src.models.classical.softmax import (
    compute_cost_function,
    compute_probabilities,
    run_gradient_descent_iteration,
    update_y,
)
from src.wrappers import svm_predict


def test_get_mnist_data_shapes() -> None:
    """Return arrays with consistent shapes and valid labels."""
    train_features, train_labels, test_features, test_labels = get_mnist_data()
    assert train_features.shape[0] == train_labels.shape[0]
    assert test_features.shape[0] == test_labels.shape[0]
    assert train_features.shape[1] == 784
    assert test_features.shape[1] == 784
    assert train_features.shape[0] > 0
    assert test_features.shape[0] > 0
    assert set(np.unique(train_labels)).issubset(set(range(10)))


def test_closed_form_reference_solution() -> None:
    """Match the reference closed-form ridge regression solution."""
    design_matrix = np.arange(1, 16, dtype=float).reshape(3, 5)
    targets = np.arange(1, 4, dtype=float)
    lambda_factor = 0.5
    expected_parameters = np.array(
        [-0.03411225, 0.00320187, 0.04051599, 0.07783012, 0.11514424],
        dtype=float,
    )
    parameters = closed_form(design_matrix, targets, lambda_factor)
    npt.assert_allclose(parameters, expected_parameters, rtol=1e-6, atol=1e-6)


def test_one_vs_rest_svm_linearly_separable() -> None:
    """Classify a linearly separable dataset with a linear SVM."""
    train_features = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    train_labels = np.array([0.0, 0.0, 1.0, 1.0])
    test_features = np.array(
        [
            [0.2, 0.1],
            [0.1, 0.9],
            [0.9, 0.2],
            [0.8, 0.9],
        ]
    )
    expected = np.array([0.0, 0.0, 1.0, 1.0])
    predicted = svm_predict(train_features, train_labels, test_features, random_state=0, C=1000.0, kernel="linear")
    npt.assert_allclose(predicted, expected, rtol=0, atol=0)


def test_compute_probabilities_uniform_and_peaked() -> None:
    """Return uniform probabilities for zero parameters and peaked for monotonic parameters."""
    num_samples = 3
    num_features = 5
    num_classes = 7
    features = np.arange(num_samples * num_features, dtype=float).reshape(num_samples, num_features)
    parameters = np.zeros((num_classes, num_features), dtype=float)
    temperature = 0.2
    expected_uniform = np.full((num_classes, num_samples), 1.0 / num_classes)
    probabilities = compute_probabilities(features, parameters, temperature)
    npt.assert_allclose(probabilities, expected_uniform, rtol=1e-6, atol=1e-6)

    parameters = np.arange(num_classes * num_features, dtype=float).reshape(num_classes, num_features)
    probabilities = compute_probabilities(features, parameters, temperature)
    predicted_labels = np.argmax(probabilities, axis=0)
    expected_labels = np.full(num_samples, num_classes - 1)
    npt.assert_array_equal(predicted_labels, expected_labels)


def test_compute_cost_function_reference() -> None:
    """Match the reference softmax cost value."""
    num_samples = 3
    num_features = 5
    num_classes = 7
    features = np.arange(num_samples * num_features, dtype=float).reshape(num_samples, num_features)
    labels = np.arange(num_samples, dtype=int)
    parameters = np.zeros((num_classes, num_features), dtype=float)
    temperature = 0.2
    lambda_factor = 0.5
    expected_cost = 1.9459101490553135
    cost_value = compute_cost_function(features, labels, parameters, lambda_factor, temperature)
    npt.assert_allclose(cost_value, expected_cost, rtol=1e-6, atol=1e-6)


def test_run_gradient_descent_iteration_reference() -> None:
    """Match the reference gradient descent update."""
    num_samples = 3
    num_features = 5
    num_classes = 7
    features = np.arange(num_samples * num_features, dtype=float).reshape(num_samples, num_features)
    labels = np.arange(num_samples, dtype=int)
    parameters = np.zeros((num_classes, num_features), dtype=float)
    step_size = 2.0
    temperature = 0.2
    lambda_factor = 0.5
    expected_parameters = np.array(
        [
            [-7.14285714, -5.23809524, -3.33333333, -1.42857143, 0.47619048],
            [9.52380952, 11.42857143, 13.33333333, 15.23809524, 17.14285714],
            [26.19047619, 28.0952381, 30.0, 31.9047619, 33.80952381],
            [-7.14285714, -8.57142857, -10.0, -11.42857143, -12.85714286],
            [-7.14285714, -8.57142857, -10.0, -11.42857143, -12.85714286],
            [-7.14285714, -8.57142857, -10.0, -11.42857143, -12.85714286],
            [-7.14285714, -8.57142857, -10.0, -11.42857143, -12.85714286],
        ]
    )
    updated_parameters = run_gradient_descent_iteration(
        features,
        labels,
        parameters,
        step_size,
        lambda_factor,
        temperature,
    )
    npt.assert_allclose(updated_parameters, expected_parameters, rtol=1e-6, atol=1e-6)


def test_update_y_mod3() -> None:
    """Convert labels to their mod-3 equivalents."""
    train_labels = np.arange(0, 10, dtype=int)
    test_labels = np.arange(9, -1, -1, dtype=int)
    expected_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=int)
    expected_test = np.array([0, 2, 1, 0, 2, 1, 0, 2, 1, 0], dtype=int)
    mod_train, mod_test = update_y(train_labels, test_labels)
    npt.assert_array_equal(mod_train, expected_train)
    npt.assert_array_equal(mod_test, expected_test)


def test_project_onto_pc_reference() -> None:
    """Match the reference PCA projection values."""
    features = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ]
    )
    centered_features, feature_means = center_data(features)
    principal_components_matrix = principal_components(centered_features)
    expected_projection = np.array(
        [
            [-5.61248608, 0.0],
            [-1.87082869, 0.0],
            [1.87082869, 0.0],
            [5.61248608, 0.0],
        ]
    )
    projection = project_onto_PC(features, principal_components_matrix, n_components=2, feature_means=feature_means)
    npt.assert_allclose(projection, expected_projection, rtol=1e-6, atol=1e-6)


def test_cubic_features_1d_reference() -> None:
    """Match the reference cubic feature mapping for 1D inputs."""
    features = np.array([[np.sqrt(3.0)], [0.0]])
    cubic = np.sort(cubic_features(features))
    expected = np.array(
        [
            [1.0, np.sqrt(9.0), np.sqrt(27.0), np.sqrt(27.0)],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    npt.assert_allclose(cubic, expected, rtol=1e-6, atol=1e-6)


def test_cubic_features_2d_reference() -> None:
    """Match the reference cubic feature mapping for symmetric 2D inputs."""
    features = np.array([[np.sqrt(3.0), np.sqrt(3.0)], [0.0, 0.0]])
    cubic = np.sort(cubic_features(features))
    expected = np.array(
        [
            [
                1.0,
                3.0,
                3.0,
                5.19615242,
                5.19615242,
                5.19615242,
                5.19615242,
                7.34846923,
                9.0,
                9.0,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    npt.assert_allclose(cubic, expected, rtol=1e-6, atol=1e-6)


def test_cubic_features_2d_asymmetric_reference() -> None:
    """Match the reference cubic feature mapping for asymmetric 2D inputs."""
    features = np.array([[np.sqrt(3.0), 0.0], [0.0, np.sqrt(3.0)]])
    cubic = np.sort(cubic_features(features))
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 5.19615242, 5.19615242],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 5.19615242, 5.19615242],
        ]
    )
    npt.assert_allclose(cubic, expected, rtol=1e-6, atol=1e-6)


def test_polynomial_kernel_matches_definition() -> None:
    """Compute the polynomial kernel exactly."""
    features_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    features_b = np.array([[0.5, -1.0], [2.0, 1.5], [0.0, 3.0]])
    coefficient = 1.0
    degree = 2
    kernel_matrix = polynomial_kernel(features_a, features_b, c=coefficient, p=degree)
    expected_kernel = (features_a @ features_b.T + coefficient) ** degree
    npt.assert_allclose(kernel_matrix, expected_kernel, rtol=1e-6, atol=1e-6)


def test_rbf_kernel_matches_definition() -> None:
    """Compute the RBF kernel exactly."""
    features_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    features_b = np.array([[0.5, -1.0], [2.0, 1.5], [0.0, 3.0]])
    gamma = 0.5
    kernel_matrix = rbf_kernel(features_a, features_b, gamma=gamma)
    squared_norms_a = np.sum(features_a**2, axis=1).reshape(-1, 1)
    squared_norms_b = np.sum(features_b**2, axis=1).reshape(1, -1)
    squared_distances = squared_norms_a + squared_norms_b - 2 * (features_a @ features_b.T)
    expected_kernel = np.exp(-gamma * squared_distances)
    npt.assert_allclose(kernel_matrix, expected_kernel, rtol=1e-6, atol=1e-6)
