# Explore linear algebra basics to understand the mathematical aspects of ML


### 1. Scalars, Vectors, and Matrices:
- **Scalars:** Single numerical value (e.g., $$a = 5$$).
- **Vectors (1D array):** A collection of scalars represented vertically or horizontally (e.g., $$mathbf{v} = [1, 2, 3]$$).
- **Matrices (2D array):** An arrangement of scalars in rows and columns (e.g., $$mathbf{A} = $$begin{bmatrix} 1 & 2 $$$$ 3 & 4 $$end{bmatrix}$$).

### 2. Matrix Operations:
- **Matrix Addition:** If $$mathbf{A}$$ and $$mathbf{B}$$ are matrices of the same size, their sum is obtained by adding corresponding elements: $$mathbf{C} = $$mathbf{A} + $$mathbf{B}$$.
- **Matrix Multiplication:** The dot product of matrices $$mathbf{A}$$ ($$m $$times n$$) and $$mathbf{B}$$ ($$n $$times p$$) results in a matrix $$mathbf{C}$$ ($$m $$times p$$): $$mathbf{C}_{ij} = $$sum_{k=1}^n $$mathbf{A}_{ik} $$times $$mathbf{B}_{kj}$$.

### 3. Matrix Transposition:
- **Transpose of a Matrix:** The transpose of a matrix $$mathbf{A}$$ is denoted as $$mathbf{A}^T$$ and is obtained by flipping the elements across the main diagonal.

### 4. Matrix Inverse and Pseudoinverse:
- **Matrix Inverse:** For a square matrix $$mathbf{A}$$, if $$mathbf{A}$$mathbf{A}^{-1} = $$mathbf{A}^{-1}$$mathbf{A} = $$mathbf{I}$$ (identity matrix), then $$mathbf{A}^{-1}$$ is the inverse of $$mathbf{A}$$.
- **Pseudoinverse:** For a non-square matrix, the pseudoinverse $$mathbf{A}^+$$ allows solving linear equations.

### 5. Linear Dependence and Independence:
- **Linear Dependence:** Vectors are linearly dependent if one vector can be expressed as a linear combination of the others.
- **Linear Independence:** Vectors are linearly independent if no vector can be expressed as a linear combination of the others.

### 6. Norms:
- **$$L_p$$ Norms:** $$|$$mathbf{x}$$|_p = $$left($$sum_{i=1}^n |x_i|^p$$right)^{$$frac{1}{p}}$$.
- **Euclidean Norm ($$L_2$$ Norm):** $$|$$mathbf{x}$$|_2 = $$sqrt{$$sum_{i=1}^n x_i^2}$$.

### 7. Eigenvalues and Eigenvectors:
- **Eigenvalue ($$lambda$$) and Eigenvector ($$mathbf{v}$$):** For a square matrix $$mathbf{A}$$, $$mathbf{A}$$mathbf{v} = $$lambda$$mathbf{v}$$, where $$mathbf{v}$$ is the eigenvector and $$lambda$$ is the eigenvalue.

### 8. Solving Linear Systems:
- **Matrix Equation:** $$mathbf{Ax} = $$mathbf{b}$$, where $$mathbf{A}$$ is a matrix, $$mathbf{x}$$ is the unknown vector, and $$mathbf{b}$$ is the known vector.
- **Gaussian Elimination:** Row operations to transform $$mathbf{A}$$mathbf{x} = $$mathbf{b}$$ into row-echelon form and then solve.

### 9. Matrix Decompositions:
- **Singular Value Decomposition (SVD):** $$mathbf{A} = $$mathbf{U} $$Sigma $$mathbf{V}^T$$, where $$mathbf{U}$$ and $$mathbf{V}$$ are orthogonal matrices, and $$Sigma$$ is a diagonal matrix of singular values.

### 10. Vector Spaces and Subspaces:
- **Vector Space:** A set of vectors that is closed under vector addition and scalar multiplication.
- **Subspace:** A subset of a vector space that is also a vector space.

### 11. Orthogonality:
- **Orthogonal Vectors:** Two vectors are orthogonal if their dot product is zero.
- **Orthogonal Projection:** Projecting a vector onto another in a way that they are orthogonal.

### 12. Determinants:
- **Determinant of a Matrix ($$det($$mathbf{A})$$):** A scalar value calculated from the elements of a square matrix.

### 13. Special Matrices:
- **Identity Matrix ($$mathbf{I}$$):** A square matrix with ones on the main diagonal and zeros elsewhere.
- **Diagonal Matrix:** A matrix where non-diagonal elements are zero.
- **Symmetric Matrix:** A square matrix that is equal to its transpose.



### Test Case 1 (Example):
Let's consider a matrix multiplication example:
- Matrix $$mathbf{A} = $$begin{bmatrix} 1 & 2 $$$$ 3 & 4 $$end{bmatrix}$$.
- Matrix $$mathbf{B} = $$begin{bmatrix} 5 & 6 $$$$ 7 & 8 $$end{bmatrix}$$.

We want to calculate $$mathbf{C} = $$mathbf{A} $$times $$mathbf{B}$$:
- $$mathbf{C} = $$begin{bmatrix} 1 $$times 5 + 2 $$times 7 & 1 $$times 6 + 2 $$times 8 $$$$ 3 $$times 5 + 4 $$times 7 & 3 $$times 6 + 4 $$times 8 $$end{bmatrix}$$.
- Simplifying, we get $$mathbf{C} = $$begin{bmatrix} 19 & 22 $$$$ 43 & 50 $$end{bmatrix}$$.

### 14. Matrix Rank:
- **Rank of a Matrix ($$ $$text{rank}($$mathbf{A}) $$):** The maximum number of linearly independent rows or columns in a matrix.

### 15. Sparse Matrices:
- **Sparse Matrix:** A matrix in which most of the elements are zero.
- **Compressed Sparse Row (CSR) Format:** A popular format for storing sparse matrices efficiently.

### 16. Inner Product and Orthogonality:
- **Inner Product (Dot Product):** $$ $$mathbf{u} $$cdot $$mathbf{v} = $$sum_{i=1}^n u_i v_i $$.
- **Orthogonal Vectors:** $$ $$mathbf{u} $$ and $$ $$mathbf{v} $$ are orthogonal if $$ $$mathbf{u} $$cdot $$mathbf{v} = 0 $$.

### 17. Hyperplanes and Linear Equations:
- **Hyperplane:** A subspace with one dimension less than its ambient space.
- **Linear Equation:** $$ a_1x_1 + a_2x_2 + $$ldots + a_nx_n = b $$.



### Test Case (Example 2):
Let's calculate the inverse of a 2x2 matrix:
- Matrix $$ $$mathbf{A} = $$begin{bmatrix} 2 & 3 $$$$ 1 & 4 $$end{bmatrix} $$.

To find the inverse $$ $$mathbf{A}^{-1} $$, we use the formula:
$$ $$mathbf{A}^{-1} = $$frac{1}{ad-bc} $$begin{bmatrix} d & -b $$$$ -c & a $$end{bmatrix} $$
where $$ a, b, c, $$ and $$ d $$ are elements of $$ $$mathbf{A} $$, and $$ ad - bc $$ is the determinant.

- Calculate the determinant: $$ ad - bc = (2 $$times 4) - (3 $$times 1) = 8 - 3 = 5 $$.
- Calculate $$ $$mathbf{A}^{-1} $$:
$$ $$mathbf{A}^{-1} = $$frac{1}{5} $$begin{bmatrix} 4 & -3 $$$$ -1 & 2 $$end{bmatrix} = $$begin{bmatrix} 0.8 & -0.6 $$$$ -0.2 & 0.4 $$end{bmatrix} $$

This example demonstrates finding the inverse of a 2x2 matrix, a fundamental operation used in various ML algorithms.

