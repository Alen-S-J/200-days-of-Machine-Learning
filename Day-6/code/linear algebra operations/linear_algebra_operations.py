import numpy as np

# Matrix multiplication
# Matrix multiplication is a fundamental operation in linear algebra.
# It's used extensively in various machine learning algorithms, including neural networks.
A = np.array([[1, 2], [3, 4]])  # Define a 2x2 matrix A
B = np.array([[5, 6], [7, 8]])  # Define a 2x2 matrix B
C = np.dot(A, B)  # Multiply matrices A and B using dot product
print("Matrix Multiplication Result:\n", C)

# Vector operations
# Vectors are a fundamental data structure in linear algebra and machine learning.
v1 = np.array([1, 2, 3])  # Define a 1D array (vector) v1
v2 = np.array([4, 5, 6])  # Define another 1D array (vector) v2
dot_product = np.dot(v1, v2)  # Calculate dot product of vectors v1 and v2
print("Dot Product of v1 and v2:", dot_product)

# Solving a system of linear equations
# Solving linear equations is a crucial part of many machine learning algorithms, particularly in regression.
# It helps find the parameters that best fit the given data.
coefficients = np.array([[2, 3], [4, -5]])  # Coefficients matrix
constants = np.array([8, -1])  # Constants vector
solution = np.linalg.solve(coefficients, constants)  # Solve the linear system
print("Solution to the system of linear equations:", solution)

# Eigenvalues and eigenvectors
# Eigenvalues and eigenvectors have various applications, including dimensionality reduction in machine learning (e.g., PCA).
A = np.array([[4, -2], [1, 1]])  # Define a 2x2 matrix A
eigenvalues, eigenvectors = np.linalg.eig(A)  # Compute eigenvalues and eigenvectors of A
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Singular Value Decomposition (SVD)
# SVD is a critical technique used in dimensionality reduction, collaborative filtering, and more.
A = np.array([[1, 2], [3, 4], [5, 6]])  # Define a 3x2 matrix A
U, S, VT = np.linalg.svd(A)  # Perform Singular Value Decomposition
print("U:\n", U)  # Left singular vectors
print("S:\n", S)  # Singular values
print("VT:\n", VT)  # Right singular vectors

