# Dynamic Time Warping

Fast implementation of Dynamic Time Warping algorithm using numba for just in time compilation into machine code. 

This package allows to calculate the distance between two 1D sequences using the DTW algorithm, or all the pairewise distances between sequences in a 2D array.

# Installation

Using pip:
```
pip install git+https://github.com/clabrugere/dynamic-time-warping.git
```

# Usage

```
import numpy as np
from dtw import dtw, pairwise

X = np.random.rand(100, 1000).astype(np.float64)
x, y = X[0, :], X[1, :]

# between two 1d arrays
d = dtw(x, y, window=100, normalize=False)

# between several 1d arrays
D = pairwise(X, window=100, normalize=False)
```

Notes:
 - The implementation doesn't return the full alignement matrix but only keeps track of two 1D arrays in order to reduce the space complexity of the algorithm to O(N).
 - input arrays must be numpy arrays of type float64 and contiguous (order="C").
 - the `window` argument adds a locality constraint and reduces the number of distance evaluations when used. Default to 0 (no constraint).

# License

[MIT](LICENSE)