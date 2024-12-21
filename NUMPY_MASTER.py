import numpy as np

# ====================================
# 1. Creating a NumPy array
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)

# Shape and Size
print("Shape:", arr.shape)   # (5,)
print("Size:", arr.size)     # 5

# Reshaping
arr_2d = arr.reshape(1, 5)
print("Reshaped to 2D:", arr_2d)

# ====================================
# 2. Argmax Example
arr = np.array([10, 20, 5, 40, 30])

# Index of max value
index = np.argmax(arr)
print("Argmax Index:", index)  # 3 (index of 40)

# For 2D arrays
arr_2d = np.array([[10, 20], [30, 40]])
index_col = np.argmax(arr_2d, axis=0)  # Max index column-wise
index_row = np.argmax(arr_2d, axis=1)  # Max index row-wise
print("Argmax Column-wise:", index_col)  # [1 1]
print("Argmax Row-wise:", index_row)     # [1 1]

# ====================================
# 3. Softmax Calculation
arr = np.array([2.0, 1.0, 0.1])

# Calculate softmax
exp_values = np.exp(arr)
softmax = exp_values / np.sum(exp_values)
print("Softmax:", softmax)  # [0.65900114 0.24243297 0.09856589]

# ====================================
# 4. Normalization Example
arr = np.array([1, 2, 3, 4, 5])
normalized = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
print("Normalized Array:", normalized)  # [0.   0.25 0.5  0.75 1. ]

# ====================================
# 5. Dot Product and Matrix Multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Dot Product
result = np.dot(A, B)
print("Dot Product:\n", result)
# [[19 22]
#  [43 50]]

# ====================================
# 6. Log-Sum-Exp (Numerical Stability in Softmax)
arr = np.array([1000, 2000, 3000])

# Numerical stability trick
max_val = np.max(arr)
stable_softmax = np.exp(arr - max_val) / np.sum(np.exp(arr - max_val))
print("Stable Softmax:", stable_softmax)  # [0. 0. 1.]

# ====================================
# 7. Top-k Values
arr = np.array([10, 20, 5, 40, 30])
k = 3

# Get indices of top k values
indices = np.argsort(arr)[-k:]
print("Top-k Indices:", indices)  # [1 4 3]

# Get top k values
top_k_values = arr[indices]
print("Top-k Values:", top_k_values)  # [20 30 40]

# ====================================
# 8. Top-k in Descending Order
top_k_desc = arr[np.argsort(arr)[-k:]][::-1]
print("Top-k Descending Order:", top_k_desc)  # [40 30 20]

# ====================================
# 9. Softmax with Top-2 Values
arr = np.array([3.0, 1.0, 0.2])

# Softmax
exp_vals = np.exp(arr)
softmax = exp_vals / np.sum(exp_vals)

# Top-2 values
top2_indices = np.argsort(softmax)[-2:]  # Indices
top2_probs = softmax[top2_indices]       # Probabilities

print("Softmax:", softmax)
print("Top-2 Indices:", top2_indices)
print("Top-2 Probabilities:", top2_probs)
