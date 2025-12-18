import numpy as np

arr = np.arange(1, 13).reshape(4, 3)
# print(arr.size, arr.shape, str(arr.ndim) + 'D')
arr2 = np.resize(arr, (2, 3, 8))
# print(np.sort(arr2, axis=None))

# New dimensions
vector = np.array([1, 2, 3, 4])  # 1 dim
matrix = vector[np.newaxis, :]  # Method 1
# print(matrix.ndim)

# matrix = vector.reshape(2, 2)  # Method 2
# print(matrix.ndim)

# Concatenating arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
# print(np.concatenate((b, a)).reshape(2, 3))

# Array operations
arr = np.arange(0, 10)
# print(arr[arr % 2 == 0])  # Filter even values

# print(arr.max(), arr.min(), arr.sum(), arr.mean())

rng = np.random.default_rng()
rand = rng.integers(99, size=(2, 3, 3))
print(rand)

