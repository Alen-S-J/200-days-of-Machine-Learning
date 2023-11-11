

# **K-means Clustering**

K-means is a popular unsupervised machine learning algorithm used for clustering data points into groups or clusters. It aims to partition data points into K clusters, where each point belongs to the cluster with the nearest mean value. Here are the key components and steps of K-means:

1. **Initialization**: Choose K initial cluster centroids. This can be done randomly or using more advanced techniques. These centroids will represent the initial cluster centers.

2. **Assignment**: For each data point, calculate the distance to each of the K centroids and assign the point to the cluster associated with the nearest centroid. The most common distance metric is the Euclidean distance.

3. **Update Centroids**: Recalculate the centroids of each cluster as the mean of all the data points assigned to that cluster.

4. **Repeat**: Steps 2 and 3 are repeated iteratively until convergence. Convergence occurs when the centroids no longer change significantly, or a specified number of iterations is reached.

5. **Final Clustering**: After convergence, the data points are clustered into K groups, and each point belongs to the cluster with the nearest centroid.

#### **Objective Function - Sum of Squared Distances (Inertia)**

The objective of K-means clustering is to minimize the sum of squared distances (inertia) within each cluster. Mathematically, the objective function is defined as:

$$J = $$sum_{i=1}^{K} $$sum_{x $$in C_i} ||x - $$mu_i||^2$$

Where:
- $$J$$ is the total sum of squared distances (inertia).
- $$K$$ is the number of clusters.
- $$C_i$$ is the i-th cluster.
- $$mu_i$$ is the centroid of the i-th cluster.
- $$||x - $$mu_i||^2$$ is the squared Euclidean distance between a data point $$x$$ and the centroid $$mu_i$$ of its assigned cluster.

Minimizing $$J$$ means that data points are closer to their respective cluster centroids.

### **Use Case Scenarios**

K-means clustering is a versatile algorithm with various use case scenarios:

1. **Customer Segmentation**: Businesses can use K-means to group customers based on their purchasing behavior. This information can help in targeted marketing and product recommendations.

2. **Image Compression**: K-means can be used to compress images by clustering similar colors. Each cluster represents a color, and the image is reconstructed using these cluster centroids.

3. **Anomaly Detection**: K-means can identify data points that do not belong to any cluster. Outliers are often located far from the cluster centroids and can be detected as anomalies.

4. **Document Clustering**: K-means can group similar documents together, making it useful for organizing and categorizing large collections of text data.

5. **Recommendation Systems**: In collaborative filtering, K-means can be used to cluster users or items based on their preferences and behaviors, aiding in making recommendations.

6. **Genomic Data Analysis**: K-means can be applied to cluster genes or proteins based on their expression levels, aiding in genomics research.

7. **Retail Inventory Management**: Clustering products in a store based on sales data can help in optimizing inventory management and shelf placement.

8. **Natural Language Processing (NLP)**: K-means can be used to cluster similar documents or texts in NLP tasks, such as topic modeling and text categorization.

Remember that while K-means is a powerful tool, it has some limitations, such as sensitivity to the initial centroid selection and the need to specify the number of clusters, which may not always be straightforward in real-world applications. It's essential to understand these limitations and consider other clustering algorithms when they are more suitable for your specific problem.