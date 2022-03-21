# Dynamic Time Warping

Fast implementation of Dynamic Time Warping algorithm using numba JIT compilation. 

This package allows to calculate the distance between two sequences using DTW algorithm, or all the pairewise
distances between sequences in a 2D array.

# Installation

Using pip:
```
pip install .
```

# Usage

```
import numpy as np
from dtw import dtw, pairwise

X = np.random.randint(-10, 10, (100, 100)).astype(np.float64)

x, y = X[0, :], X[1, :]
d = dtw(x, y)

D = pairwise(X)
```

# License

[MIT](LICENSE)