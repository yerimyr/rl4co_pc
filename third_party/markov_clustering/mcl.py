"""Vendored MCL implementation from markov_clustering 0.0.6.dev.

Copyright (c) 2017 Guy Allard. MIT License.

The algorithmic functions are retained from the user-supplied GitHub archive.
The only compatibility change is that ``normalize`` has a NumPy/SciPy fallback
when scikit-learn is unavailable. This computes the same column-wise L1
normalization used by the upstream implementation.
"""

import numpy as np
from scipy.sparse import csc_matrix, diags, dok_matrix, isspmatrix

from .utils import MessagePrinter

try:  # Preserve the upstream implementation when scikit-learn is installed.
    from sklearn.preprocessing import normalize as _sklearn_normalize
except ImportError:  # pragma: no cover - this is the expected project setup.
    _sklearn_normalize = None


def sparse_allclose(a, b, rtol=1e-5, atol=1e-8):
    c = np.abs(a - b) - rtol * np.abs(b)
    return c.max() <= atol


def normalize(matrix):
    """L1-normalize matrix columns, matching sklearn axis=0 behavior."""
    if _sklearn_normalize is not None:
        return _sklearn_normalize(matrix, norm="l1", axis=0)

    if isspmatrix(matrix):
        matrix = matrix.tocsc().astype(float)
        sums = np.asarray(matrix.sum(axis=0)).reshape(-1)
        inverse = np.zeros_like(sums, dtype=float)
        np.divide(1.0, sums, out=inverse, where=sums != 0)
        return matrix @ diags(inverse)

    matrix = np.asarray(matrix, dtype=float)
    sums = matrix.sum(axis=0, keepdims=True)
    normalized = np.zeros_like(matrix, dtype=float)
    np.divide(matrix, sums, out=normalized, where=sums != 0)
    return normalized


def inflate(matrix, power):
    if isspmatrix(matrix):
        return normalize(matrix.power(power))
    return normalize(np.power(matrix, power))


def expand(matrix, power):
    if isspmatrix(matrix):
        return matrix**power
    return np.linalg.matrix_power(matrix, power)


def add_self_loops(matrix, loop_value):
    shape = matrix.shape
    assert shape[0] == shape[1], "Error, matrix is not square"

    new_matrix = matrix.todok() if isspmatrix(matrix) else matrix.copy()
    for i in range(shape[0]):
        new_matrix[i, i] = loop_value
    return new_matrix.tocsc() if isspmatrix(matrix) else new_matrix


def prune(matrix, threshold):
    if isspmatrix(matrix):
        pruned = dok_matrix(matrix.shape)
        pruned[matrix >= threshold] = matrix[matrix >= threshold]
        pruned = pruned.tocsc()
    else:
        pruned = matrix.copy()
        pruned[pruned < threshold] = 0

    num_cols = matrix.shape[1]
    row_indices = matrix.argmax(axis=0).reshape((num_cols,))
    col_indices = np.arange(num_cols)
    pruned[row_indices, col_indices] = matrix[row_indices, col_indices]
    return pruned


def converged(matrix1, matrix2):
    if isspmatrix(matrix1) or isspmatrix(matrix2):
        return sparse_allclose(matrix1, matrix2)
    return np.allclose(matrix1, matrix2)


def iterate(matrix, expansion, inflation):
    matrix = expand(matrix, expansion)
    matrix = inflate(matrix, inflation)
    return matrix


def get_clusters(matrix):
    if not isspmatrix(matrix):
        matrix = csc_matrix(matrix)

    attractors = matrix.diagonal().nonzero()[0]
    clusters = set()
    for attractor in attractors:
        cluster = tuple(matrix.getrow(attractor).nonzero()[1].tolist())
        clusters.add(cluster)
    return sorted(list(clusters))


def run_mcl(
    matrix,
    expansion=2,
    inflation=2,
    loop_value=1,
    iterations=100,
    pruning_threshold=0.001,
    pruning_frequency=1,
    convergence_check_frequency=1,
    verbose=False,
):
    assert expansion > 1, "Invalid expansion parameter"
    assert inflation > 1, "Invalid inflation parameter"
    assert loop_value >= 0, "Invalid loop_value"
    assert iterations > 0, "Invalid number of iterations"
    assert pruning_threshold >= 0, "Invalid pruning_threshold"
    assert pruning_frequency > 0, "Invalid pruning_frequency"
    assert convergence_check_frequency > 0, "Invalid convergence_check_frequency"

    printer = MessagePrinter(verbose)
    printer.print("-" * 50)
    printer.print("MCL Parameters")
    printer.print("Expansion: {}".format(expansion))
    printer.print("Inflation: {}".format(inflation))
    if pruning_threshold > 0:
        printer.print(
            "Pruning threshold: {}, frequency: {} iteration{}".format(
                pruning_threshold,
                pruning_frequency,
                "s" if pruning_frequency > 1 else "",
            )
        )
    else:
        printer.print("No pruning")
    printer.print(
        "Convergence check: {} iteration{}".format(
            convergence_check_frequency,
            "s" if convergence_check_frequency > 1 else "",
        )
    )
    printer.print("Maximum iterations: {}".format(iterations))
    printer.print("{} matrix mode".format("Sparse" if isspmatrix(matrix) else "Dense"))
    printer.print("-" * 50)

    if loop_value > 0:
        matrix = add_self_loops(matrix, loop_value)
    matrix = normalize(matrix)

    for i in range(iterations):
        printer.print("Iteration {}".format(i + 1))
        last_mat = matrix.copy()
        matrix = iterate(matrix, expansion, inflation)

        if pruning_threshold > 0 and i % pruning_frequency == pruning_frequency - 1:
            printer.print("Pruning")
            matrix = prune(matrix, pruning_threshold)

        if i % convergence_check_frequency == convergence_check_frequency - 1:
            printer.print("Checking for convergence")
            if converged(matrix, last_mat):
                printer.print(
                    "Converged after {} iteration{}".format(
                        i + 1, "s" if i > 0 else ""
                    )
                )
                break

    printer.print("-" * 50)
    return matrix
