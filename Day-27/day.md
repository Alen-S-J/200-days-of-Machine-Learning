

# What is Principal Component Analysis (PCA)?

Principal Component Analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of correlated variables into uncorrelated variables. It aims to reduce the dimensionality of a dataset while preserving the most important patterns or relationships between variables. PCA is widely used in exploratory data analysis and machine learning for predictive models.

### How PCA Works

1. **Standardization**: The first step is to standardize the dataset to ensure that each variable has a mean of 0 and a standard deviation of 1.

   ```markdown
   Z = (X - μ) / σ
   ```

   - Where:
     - μ is the mean of independent features.
     - σ is the standard deviation of independent features.

2. **Covariance Matrix Computation**: The covariance matrix measures the strength of joint variability between variables, indicating how much they change in relation to each other.

   ```markdown
   cov(x1, x2) = Σ (x1_i - μ1)(x2_i - μ2) / (n - 1)
   ```

3. **Compute Eigenvalues and Eigenvectors**: Eigenvalues and eigenvectors of the covariance matrix are computed to identify the principal components. Eigenvalues represent the amount of variance, and eigenvectors represent the direction in which the data varies the most.

4. **Sort Eigenvalues and Eigenvectors**: Sort the eigenvalues in descending order and sort the corresponding eigenvectors accordingly.

5. **Determine the Number of Principal Components**: Decide on the number of principal components to retain based on explained variance, typically retaining components that explain a certain percentage of the total variance.

6. **Project Data onto Principal Components**: Find the projection matrix and project the data onto the selected principal components.

### Advantages of PCA

- Dimensionality Reduction
- Feature Selection
- Data Visualization
- Multicollinearity Handling
- Noise Reduction
- Data Compression
- Outlier Detection

### Disadvantages of PCA

- Interpretation of Principal Components
- Data Scaling
- Information Loss
- Non-linear Relationships
- Computational Complexity
- Overfitting

