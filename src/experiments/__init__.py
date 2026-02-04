"""Experiment runners and analysis helpers used by the notebook."""

from src.experiments.classical import (
    compute_pca_representations,
    run_kernelised_softmax_on_mnist_pca10,
    run_linear_regression_on_mnist,
    run_multiclass_svm_on_mnist,
    run_sklearn_svm_poly_on_pca10,
    run_sklearn_svm_rbf_on_pca10,
    run_softmax_on_mnist,
    run_softmax_on_mnist_cubic,
    run_softmax_on_mnist_eval_mod3,
    run_softmax_on_mnist_mod3,
    run_softmax_on_mnist_pca,
    run_svm_one_vs_rest_on_mnist,
)
from src.experiments.temperature import analyse_temperature_effects

__all__ = [
    "analyse_temperature_effects",
    "compute_pca_representations",
    "run_kernelised_softmax_on_mnist_pca10",
    "run_linear_regression_on_mnist",
    "run_multiclass_svm_on_mnist",
    "run_softmax_on_mnist",
    "run_softmax_on_mnist_cubic",
    "run_softmax_on_mnist_eval_mod3",
    "run_softmax_on_mnist_mod3",
    "run_softmax_on_mnist_pca",
    "run_sklearn_svm_poly_on_pca10",
    "run_sklearn_svm_rbf_on_pca10",
    "run_svm_one_vs_rest_on_mnist",
]
