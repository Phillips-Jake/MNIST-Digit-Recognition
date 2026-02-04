"""MNIST dataset helper."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]


def get_mnist_data(data_path: str | Path | None = None) -> tuple[FloatArray, IntArray, FloatArray, IntArray]:
    """
    Read MNIST data from the dataset file.

    Parameters
    ----------
    data_path : str | Path | None, default=None
        Optional path to ``mnist.npz``. If omitted, uses ``Datasets/mnist.npz``
        relative to the repository root.

    Returns
    -------
    tuple[FloatArray, IntArray, FloatArray, IntArray]
        - ``train_x``: Training images of shape (60000, 784) in [0, 1].
        - ``train_y``: Training labels of shape (60000,) with integer digit labels.
        - ``test_x``: Test images of shape (10000, 784) in [0, 1].
        - ``test_y``: Test labels of shape (10000,) with integer digit labels.
    """
    if data_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        data_path = repo_root / "Datasets" / "mnist.npz"
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"MNIST data file not found: {data_path}")

    with np.load(data_path) as data:
        train_x = np.asarray(data["train_x"], dtype=np.float32)
        train_y = np.asarray(data["train_y"], dtype=np.int64)
        test_x = np.asarray(data["test_x"], dtype=np.float32)
        test_y = np.asarray(data["test_y"], dtype=np.int64)

    return train_x, train_y, test_x, test_y
