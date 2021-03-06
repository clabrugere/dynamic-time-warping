import numpy as np
from numba import boolean, float64, int64, njit, prange


@njit(float64[::1](float64[::1]), nogil=True, cache=True)
def normalized(x):
    mu, sigma = np.mean(x), np.std(x)
    sigma = 1.0 if sigma == 0.0 else sigma

    return (x - mu) / sigma


@njit(float64(float64, float64), nogil=True, cache=True)
def sqrd_euclidean(x, y):
    return (x - y) * (x - y)


@njit(float64[::1](float64[::1], float64[::1]), nogil=True, cache=True)
def initialize_ub(x, y):
    n, m = x.shape[0], y.shape[0]
    max_len = max(n, m)

    if n > m:
        tmp = np.full((max_len,), y[-1], dtype=y.dtype)
        tmp[: max_len - m + 1] = y
        y = tmp

    if m > n:
        tmp = np.full((max_len,), x[-1], dtype=x.dtype)
        tmp[: max_len - n + 1] = x
        x = tmp

    ub_partials = np.empty((max_len,), dtype=x.dtype)
    ub_partials[-1] = sqrd_euclidean(x[-1], y[-1])

    for i in range(1, max_len):
        ub_partials[max_len - 1 - i] = ub_partials[max_len - i] + sqrd_euclidean(
            x[max_len - i - 1], y[max_len - i - 1]
        )

    return ub_partials


@njit(float64(float64[::1], float64[::1], int64, boolean), nogil=True, cache=True)
def dtw(x, y, window, normalize):
    # TODO implement pruning strategy
    """Compute the Dynamic Time Warping distance between sequence x and y.

    Args:
        x (1d np.array): First input sequence.
        y (1d np.array): Second input sequence.
        window (int): Size of the window around where to compute distances for alignement. Defaults to 0 (no window).
        normalize (bool): Either to normalize input sequences so that they have zero mean and unit variance, or not.

    Returns:
        float: Distance between x and y.
    """

    n, m = x.shape[0], y.shape[0]
    k = 0
    window = m if window == 0 else window

    cost = np.full((2 * window + 1,), np.inf, dtype=np.float64)
    cost_previous = np.full((2 * window + 1,), np.inf, dtype=np.float64)

    if normalize:
        x, y = normalized(x), normalized(y)

    cost[0] = sqrd_euclidean(x[0], y[0])

    for i in range(1, n):
        k = max(1, window - i)
        for j in range(max(1, i - window), min(i + window, m - 1)):

            cost[k] = sqrd_euclidean(x[i], y[j])

            a = np.inf if i - 1 < 0 or k + 1 > 2 * window else cost[k + 1]
            b = np.inf if j - 1 < 0 or k - 1 < 0 else cost[k - 1]
            c = np.inf if j - 1 < 0 or j - 1 < 0 else cost[k]

            cost[k] += min(a, b, c)
            k += 1

        cost, cost_previous = cost_previous, cost

    return cost[k - 1]


@njit(
    float64[:, ::1](float64[:, ::1], int64, boolean),
    parallel=True,
    nogil=True,
    cache=True,
)
def pairwise(X, window, normalize):
    """Compute pairwise Dynamic Time Warping distances between each sequences in X.

    Args:
        X (2d np.ndarray of shape (n_sequences, n_observations)): Array of sequences.
        window (int, optional): Size of the window around where to compute distances for alignement. 0 for no window.
        normalize (bool): Either to normalize input sequences so that they have zero mean and unit variance, or not.

    Returns:
        2d np.ndarray of shape (n_sequences, n_sequences): Distance matrix.
    """

    n = X.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float64)

    for i in prange(0, n):
        for j in range(i + 1, n):
            dist_matrix[i, j] = dtw(X[i], X[j], window, normalize)

    return dist_matrix + dist_matrix.T


@njit(
    float64[:, :, ::1](float64[:, :, ::1], int64, boolean),
    parallel=True,
    nogil=True,
    cache=True,
)
def pairwise_multivariate(X, window, normalize):
    """Compute pairwise Dynamic Time Warping distances between each multivariate sequences in X.

    Args:
        X (3d np.ndarray of shape (n_sequences, n_observations, n_features)): Array of sequences.
        window (int, optional): Size of the window around where to compute distances for alignement. 0 for no window.
        normalize (bool): Either to normalize input sequences so that they have zero mean and unit variance, or not.

    Returns:
        3d np.ndarray of shape (n_sequences, n_sequences, n_features): Distance matrices per features.
    """

    n, _, c = X.shape
    dist_matrix = np.zeros((n, n, c), dtype=np.float64)

    for i in range(c):
        dist_matrix[..., i] = pairwise(
            np.ascontiguousarray(X[..., i]), window, normalize
        )

    return dist_matrix
