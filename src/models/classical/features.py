"""PCA utilities and cubic feature mapping."""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]


def cubic_features(X: FloatArray) -> FloatArray:
    """
    Compute explicit cubic feature mapping.

    Parameters
    ----------
    X : FloatArray
        Input data of shape (n, d).

    Returns
    -------
    FloatArray
        Cubic feature mapping with shape (n, D), where
        D = Binom(d+3, 3) = (d+1)(d+2)(d+3)/6.

        Here, d denotes the original feature dimension. The mapping includes a constant
        (bias) term by implicitly augmenting each vector with an additional 1-valued
        coordinate, so the monomials are formed over an effective (d+1)-dimensional
        vector.
    """
    n, d = X.shape
    X_withones = np.ones((n, d + 1), dtype=float)
    X_withones[:, :-1] = X
    new_d = int((d + 1) * (d + 2) * (d + 3) / 6)

    new_data = np.empty((n, new_d), dtype=float)
    col_index = 0

    sqrt3 = np.sqrt(3.0)
    sqrt6 = np.sqrt(6.0)

    if d > 2:
        for i in range(d - 2):
            Xi = X[:, i]
            for j in range(i + 1, d - 1):
                Xij = Xi * X[:, j]
                k_idx = np.arange(j + 1, d)
                block = Xij[:, None] * X[:, k_idx]
                new_data[:, col_index : col_index + k_idx.size] = block * sqrt6
                col_index += k_idx.size

    for j in range(d + 1):
        Xj = X_withones[:, j]
        new_data[:, col_index] = Xj**3
        col_index += 1
        for k in range(j + 1, d + 1):
            Xk = X_withones[:, k]
            new_data[:, col_index] = (Xj**2) * Xk * sqrt3
            col_index += 1
            new_data[:, col_index] = Xj * (Xk**2) * sqrt3
            col_index += 1
            if k < d:
                new_data[:, col_index] = Xj * Xk * sqrt6
                col_index += 1

    return new_data


def center_data(X: FloatArray) -> tuple[FloatArray, FloatArray]:
    """
    Center columns of ``X`` to have mean 0.

    Parameters
    ----------
    X : FloatArray
        Input data of shape (n, d).

    Returns
    -------
    tuple[FloatArray, FloatArray]
        Centered data of shape (n, d) and column means of shape (d,).
    """
    feature_means = X.mean(axis=0)
    return (X - feature_means), feature_means


def principal_components(centered_data: FloatArray) -> FloatArray:
    """
    Compute PCA directions as eigenvectors of the scatter matrix.

    Parameters
    ----------
    centered_data : FloatArray
        Centered input data of shape (n, d).

    Returns
    -------
    FloatArray
        Principal component matrix of shape (d, d) sorted by descending eigenvalue.

    Notes
    -----
    The scatter matrix is defined as S = X^T X, where X is the centered data matrix.
    It is used rather than the covariance matrix as the scaling factor 1/n does not
    affect the eigenvectors and their relative ordering.
    """
    scatter_matrix = np.dot(centered_data.transpose(), centered_data)
    eigen_values, eigen_vectors = np.linalg.eigh(scatter_matrix)
    idx = eigen_values.argsort()[::-1]
    eigen_vectors = eigen_vectors[:, idx]
    return eigen_vectors


def project_onto_PC(
    X: FloatArray,
    pcs: FloatArray,
    n_components: int,
    feature_means: FloatArray,
) -> FloatArray:
    """
    Project samples onto the first ``n_components`` principal components.

    Parameters
    ----------
    X : FloatArray
        Input data of shape (n, d).
    pcs : FloatArray
        Principal component matrix of shape (d, d), with components as columns.
    n_components : int
        Number of principal components to keep.
    feature_means : FloatArray
        Feature means computed from the training set, shape (d,).

    Returns
    -------
    FloatArray
        Projected data of shape (n, n_components).
    """
    tilde_X = X - feature_means
    V = pcs[:, :n_components]
    return tilde_X @ V


def reconstruct_PC(
    x_pca: FloatArray,
    pcs: FloatArray,
    n_components: int,
    feature_means: FloatArray,
) -> FloatArray:
    """
    Reconstruct a single example from its PCA representation.

    Parameters
    ----------
    x_pca : FloatArray
        PCA representation of a single sample, shape (n_components,).
    pcs : FloatArray
        Principal component matrix of shape (d, d).
    n_components : int
        Number of components used in the PCA representation.
    feature_means : FloatArray
        Feature means of shape (d,).

    Returns
    -------
    FloatArray
        Reconstructed sample in the original feature space, shape (d,).
    """
    x_reconstructed = np.dot(x_pca, pcs[:, range(n_components)].T) + feature_means
    return x_reconstructed
