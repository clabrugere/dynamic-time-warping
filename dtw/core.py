import numpy as np
from numba import njit, prange, float64, int64


@njit(float64(float64, float64), nogil=True, cache=True)
def sqrd_euclidean(x, y):
    return (x - y) * (x - y)


@njit(float64[::1](float64[::1], float64[::1]), nogil=True, cache=True)
def initialize_ub(x, y):
    n, m = x.shape[0], y.shape[0]
    l = max(n, m)
    
    if n > m:
        tmp = np.full((l,), y[-1], dtype=y.dtype)
        tmp[:l - m + 1] = y
        y = tmp
    
    if m > n:
        tmp = np.full((l,), x[-1], dtype=x.dtype)
        tmp[:l - n + 1] = x
        x = tmp
    
    ub_partials = np.empty((l,), dtype=x.dtype)
    ub_partials[-1] = sqrd_euclidean(x[-1], y[-1])
    
    for i in range(1, l):
        ub_partials[l - 1 - i] = ub_partials[l - i] + sqrd_euclidean(x[l - i - 1], y[l - i - 1])
    
    return ub_partials


@njit(float64(float64[::1], float64[::1], int64), nogil=True, cache=True)
def dtw(x, y, window):
    # TODO implement pruning strategy
    """Compute the Dynamic Time Wraping distance between sequence x and y.

    Args:
        x (1d np.array): First input sequence.
        y (1d np.array): Second input sequence.
        window (int): Size of the window around where to compute distances for alignement. Defaults to 0 (no window).

    Returns:
        float: Distance between x and y.
    """
    
    n, m = x.shape[0], y.shape[0]
    window = m if window == 0 else window
    
    cost = np.full((2 * window + 1, ), np.inf, dtype=np.float64)
    cost_previous = np.full((2 * window + 1, ), np.inf, dtype=np.float64)
    
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


@njit(float64[:, ::1](float64[:, ::1], int64), parallel=True,  nogil=True, cache=True)
def pairwise(X, window=0):
    """Compute pairwise Dynamic Time Wraping distances between each sequences in X.

    Args:
        X (2d np.ndarray of shape (n_sequences, n_observations)): Array of sequences.
        window (int, optional): Size of the window around where to compute distances for alignement. Defaults to 0 (no window).

    Returns:
        2d np.ndarray of shape (n_sequences, n_sequences): Distance matrix.
    """
    
    n = X.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    
    for i in prange(0, n):
        for j in range(i + 1, n):
            dist_matrix[i, j] = dtw(X[i], X[j], window)
    
    return dist_matrix + dist_matrix.T
