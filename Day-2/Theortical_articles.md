## Clustering Algorithms

### K-means Clustering

1. **Initialization:**
   - Choose the number of clusters, \(K\).
   - Initialize \(K\) cluster centroids randomly or using advanced methods like K-means++.

2. **Assignment:**
   - Compute the distance (e.g., Euclidean) between each data point and all centroids.
   - Assign each data point to the nearest centroid.

3. **Update Centroids:**
   - Compute the mean of all data points in each cluster to update the centroids.

4. **Repeat:**
   - Iterate steps 2 and 3 until convergence (e.g., when centroids remain unchanged or a maximum number of iterations is reached).

### Hierarchical Clustering

1. **Initialization:**
   - Treat each data point as a single cluster.

2. **Compute Pairwise Distances:**
   - Calculate the distances (e.g., Euclidean) between all pairs of clusters.

3. **Merge or Split:**
   - Merge the two closest clusters (agglomerative) or split a cluster (divisive).

4. **Update Distance Matrix:**
   - Recalculate distances between the merged/split cluster and other clusters.

5. **Repeat:**
   - Iterate steps 2-4 until there is only one cluster (agglomerative) or each data point is in its own cluster (divisive).

## Dimensionality Reduction Algorithms

### Principal Component Analysis (PCA)

1. **Standardization:**
   - Standardize the features by scaling to mean 0 and variance 1.

2. **Covariance Matrix:**
   - Compute the covariance matrix of the standardized data.

3. **Eigenvalue Decomposition:**
   - Compute the eigenvectors and eigenvalues of the covariance matrix.

4. **Sort Eigenvalues:**
   - Sort eigenvalues in descending order and their corresponding eigenvectors.

5. **Projection Matrix:**
   - Form the projection matrix by selecting the top \(k\) eigenvectors.

6. **Projection:**
   - Project the original data onto the lower-dimensional subspace using the projection matrix.

## Association Algorithms

### Apriori Algorithm

1. **Generate Candidate Itemsets:**
   - Start by identifying frequent individual items (singletons) in the dataset.

2. **Join Step:**
   - Generate candidate itemsets of length \(k\) by joining frequent itemsets of length \(k-1\).

3. **Prune Step:**
   - Eliminate candidate itemsets that contain subsets which are not frequent.

4. **Calculate Support:**
   - Count the support (occurrences) of each candidate itemset in the dataset.

5. **Generate Association Rules:**
   - Generate association rules from the frequent itemsets, calculating confidence and support for each rule.
