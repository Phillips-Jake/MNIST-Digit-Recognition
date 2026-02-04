# MNIST Digit Recognition: A Machine Learning Case Study

In `Digit Recognition.ipynb`, we present a comparative study of classical supervised learning methods for handwritten digit recognition on MNIST, from linear baselines through kernel methods.

The accompanying `src/` module contains the core training and inference code written directly with NumPy; SVM baselines are
provided via scikit-learn wrappers.

## Objective

Provide a case study of classical digit classifiers on MNIST while keeping the optimisation and modelling details explicit. Concretely, the notebook:

- Builds a progression of increasingly well-specified models (linear regression -> large-margin classifiers -> softmax regression -> PCA + feature maps -> kernels).
- Examines how the objective function, representation, and regularisation choices affect generalisation.
- Uses local libraries as the implementation for the notebook experiments.

## Dataset

The notebook loads the MNIST (Modified National Institute of Standards and Technology) dataset from `Datasets/mnist.npz` (available from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/))

## Contents

The notebook follows the progression:

1. Ridge regression baseline (MSE on digit IDs)
2. Linear SVMs (binary one-vs-rest and multiclass)
3. Multinomial (softmax) logistic regression trained by batch gradient descent
4. Temperature scaling analysis (accuracy, parameter norms, predictive entropy)
5. PCA for dimensionality reduction (including reconstructions)
6. Polynomial (cubic) feature mapping on PCA features
7. Kernel methods (polynomial and RBF)
8. Kernelised softmax via Gram-matrix representations (demonstration on a subset due to $O(n^2)$ scaling)

## Models Implemented

The core models live under `src/models/classical/` (NumPy-first). We implement the following:

- **Ridge regression (closed form)**: baseline regressor with digit rounding (`src/models/classical/linear_regression.py`)
- **Multinomial logistic regression (softmax)**: batch gradient descent with $L_2$ regularisation and temperature parameter (`src/models/classical/softmax.py`)
- **PCA utilities**: centering, PCA directions, projection + reconstruction (`src/models/classical/features.py`)
- **Explicit cubic feature mapping**: polynomial feature expansion used after PCA (`src/models/classical/features.py`)
- **Kernel methods**: polynomial and RBF kernel matrices (`src/models/classical/kernel.py`)
- **Kernelised softmax (Gram-matrix form)**: uses softmax regression on precomputed kernel matrices (experiment runners in `src/experiments/`)

Additionally, SVM baselines are wrapped from scikit-learn. In particular, these are

- **Linear / kernel SVM baselines**: wrappers around `LinearSVC` and `SVC` used both before and after PCA (`src/wrappers/sklearn_svm.py`)

## Project Structure

```
.
├── Digit Recognition.ipynb        # Main analysis notebook
├── Datasets/
│   └── mnist.npz                  # MNIST dataset (NumPy archive)
├── src/                           # Project code
│   ├── data/                      # Dataset loading + image plotting helpers
│   ├── experiments/               # High-level experiment runners + tuning scripts
│   ├── models/
│   │   └── classical/             # NumPy implementations: ridge, softmax, PCA, kernels
│   ├── visualisation/             # Plotting utilities (e.g., PCA scatter)
│   └── wrappers/                  # scikit-learn baselines (SVM wrappers)
├── tests/                         # Pytest unit/regression tests for core methods
├── environment.yml                # Environment
└── pyproject.toml                 # Project configuration
```

## Results

All results below are computed on the MNIST test set (10,000 images) loaded from `Datasets/mnist.npz`. Error rate is the
fraction of misclassified examples; accuracy is included for convenience (`accuracy = 1 - error rate`). For exact
hyperparameters and training curves, see `Digit Recognition.ipynb`.

| Method                                       | Error Rate | Accuracy |
| -------------------------------------------- | ---------: | -------: |
| Ridge regression (rounded baseline)          |     76.97% |   23.03% |
| Linear SVM (binary, 0 vs rest)               |      0.75% |   99.25% |
| Linear SVM (multiclass, one-vs-rest)         |      8.19% |   91.81% |
| Softmax regression (pixels)                  |     10.05% |   89.95% |
| Softmax regression (PCA-18)                  |     14.74% |   85.26% |
| Softmax regression (PCA-10 + cubic features) |      8.40% |   91.60% |
| SVM (polynomial kernel, PCA-10)              |      7.34% |   92.66% |
| SVM (RBF kernel, PCA-10)                     |      6.36% |   93.64% |
| Kernel softmax (RBF, PCA-10, subset n=12k)   |      8.93% |   91.07% |

### Notes

- Ridge regression (rounded baseline): uses squared loss on digit IDs and rounds/clips predictions at inference time (a deliberately misspecified baseline).
- PCA-based methods: learn PCA directions on the training set and project both train/test splits into the chosen low-dimensional representation.
- Kernel softmax (subset): trains softmax on a Gram-matrix representation built from a class-balanced subset of the PCA-10 training set; this scales as $O(n^2)$ in time/memory with the subset size.
