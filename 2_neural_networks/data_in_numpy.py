import numpy as np

# Scalar
s = np.array(5)
s.shape
x = s + 3

# Vector
v = np.array([1, 2, 3])

# Matrix
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Tensor
t = np.array([
    [
        [[1], [2]],
        [[3], [4]],
        [[5], [6]]
    ],
    [
        [[7], [8]],
        [[9], [10]],
        [[11], [12]]
    ],
    [
        [[13], [14]],
        [[15], [16]],
        [[17], [17]]
    ]
])

# Reshape
v.reshape(3, 1)
v[None, :]
v[:, None]
