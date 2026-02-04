"""Scikit-learn SVM wrappers used by experiment runners."""

from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from sklearn.svm import SVC, LinearSVC

SVMKernel: TypeAlias = Literal["linear", "poly", "rbf", "sigmoid", "precomputed"]

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]
NumberArray = npt.NDArray[np.number]


def svm_predict(
    train_x: FloatArray,
    train_y: NumberArray,
    test_x: FloatArray,
    *,
    random_state: int = 0,
    C: float = 0.1,
    kernel: SVMKernel = "linear",
    degree: int | None = None,
    gamma: str | float | None = None,
) -> NumberArray:
    """
    Train an SVM classifier and predict on ``test_x``.

    This wrapper supports both binary and multiclass classification.
    For ``kernel="linear"`` it uses ``LinearSVC`` (one-vs-rest for multiclass); otherwise it uses ``SVC``
    (one-vs-one for multiclass).

    Parameters
    ----------
    train_x : FloatArray
        Training features of shape (n, d).
    train_y : NumberArray
        Labels of shape (n,). May be binary or multiclass.
    test_x : FloatArray
        Test features of shape (m, d).
    random_state : int, default=0
        Random seed for sklearn's LinearSVC.
    C : float, default=0.1
        Regularisation parameter.
    kernel : str, default="linear"
        Kernel type. Use "linear" for ``LinearSVC``, otherwise uses ``SVC``.
    degree : int | None, default=None
        Degree for polynomial kernel (only used when ``kernel="poly"``).
        If ``None`` and ``kernel="poly"``, defaults to 3.
    gamma : str | float | None, default=None
        Kernel coefficient for "rbf", "poly" and "sigmoid" (only used for non-linear kernels).
        If ``None`` for a non-linear kernel, defaults to sklearn's common "scale".

    Returns
    -------
    NumberArray
        Predicted labels of shape (m,).
    """
    if kernel == "linear":
        clf = LinearSVC(random_state=random_state, C=C)
    else:
        if gamma is None:
            gamma = "scale"
        if kernel == "poly" and degree is None:
            degree = 3

        gamma_param: Any = gamma
        clf = SVC(
            random_state=random_state,
            kernel=kernel,
            C=C,
            degree=3 if degree is None else degree,
            gamma=gamma_param,
        )
    clf.fit(train_x, train_y)
    return clf.predict(test_x)


def compute_test_error_svm(test_y: NumberArray, pred_test_y: NumberArray) -> float:
    """
    Compute the fraction of misclassified examples.

    Parameters
    ----------
    test_y : NumberArray
        Ground-truth labels of shape (m,).
    pred_test_y : NumberArray
        Predicted labels of shape (m,).

    Returns
    -------
    float
        Fraction of misclassified examples.
    """
    return float(1 - np.mean(pred_test_y == test_y))
