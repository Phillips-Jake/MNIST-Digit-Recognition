"""Scikit-learn wrappers."""

from src.wrappers.sklearn_svm import compute_test_error_svm, svm_predict

__all__ = [
    "compute_test_error_svm",
    "svm_predict",
]
