"""Classical (NumPy-first) models and feature transforms."""

from src.models.classical.features import (
    center_data,
    cubic_features,
    principal_components,
    project_onto_PC,
    reconstruct_PC,
)
from src.models.classical.kernel import polynomial_kernel, rbf_kernel
from src.models.classical.linear_regression import closed_form, compute_test_error_linear
from src.models.classical.softmax import (
    augment_feature_vector,
    compute_cost_function,
    compute_probabilities,
    compute_test_error,
    compute_test_error_mod3,
    get_classification,
    run_gradient_descent_iteration,
    softmax_regression,
    update_y,
)

__all__ = [
    "augment_feature_vector",
    "center_data",
    "closed_form",
    "compute_cost_function",
    "compute_probabilities",
    "compute_test_error",
    "compute_test_error_linear",
    "compute_test_error_mod3",
    "cubic_features",
    "get_classification",
    "polynomial_kernel",
    "principal_components",
    "project_onto_PC",
    "rbf_kernel",
    "reconstruct_PC",
    "run_gradient_descent_iteration",
    "softmax_regression",
    "update_y",
]
