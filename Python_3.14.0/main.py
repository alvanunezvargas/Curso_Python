import numpy as np
A = np.array([
        [1, -2, -1, 3],
        [-1, 3, -2, -2],
        [2, 0, 1, 1],
        [1, -2, 2, 3]
    ], dtype=float)
X = A[1] - A[1,0] * A[0]
print(X)