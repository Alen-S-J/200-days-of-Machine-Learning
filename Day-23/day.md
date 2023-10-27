# Hierarchical clustering

Hierarchical clustering is a popular technique for grouping data into a hierarchical structure of clusters. It doesn't require specifying the number of clusters in advance, making it a flexible approach for exploring the underlying structure in data. In hierarchical clustering, data points are successively grouped into clusters, forming a tree-like structure called a dendrogram. This process allows you to explore clusters at different levels of granularity.

### Mathematical Foundations of Hierarchical Clustering:

1. **Linkage Criteria**:
   - **Single Linkage**: It measures the similarity between two clusters by the minimum distance between their data points. In other words, it considers the closest pair of data points, one from each cluster. This can lead to chain-like clusters.
   
   - **Complete Linkage**: It measures the similarity between two clusters by the maximum distance between their data points. It considers the farthest pair of data points, one from each cluster. This tends to create compact, spherical clusters.
   
   - **Average Linkage**: It calculates the similarity between clusters as the average distance between all pairs of data points, one from each cluster. This method is more balanced compared to single and complete linkage.
   
   - **Ward's Linkage**: Ward's method minimizes the increase in variance when merging clusters. It tends to produce equally sized, spherical clusters.

2. **Dendrogram**: A dendrogram is a tree-like diagram that shows the hierarchy of clusters. It is created by recursively merging clusters based on the linkage criteria. The vertical lines in the dendrogram represent clusters, and the height at which two clusters merge indicates their dissimilarity.

### Use Case Scenario:

Hierarchical clustering can be applied in various fields, including biology, social sciences, and marketing, among others. Here are a few use case scenarios:

1. **Biological Taxonomy**: Hierarchical clustering can be used to organize species into a hierarchical taxonomy based on genetic or morphological similarities. This helps biologists understand the evolutionary relationships between different species.

2. **Customer Segmentation**: In marketing, hierarchical clustering can group customers into segments based on their purchase behavior or preferences. Marketers can then tailor marketing strategies to these segments.

3. **Text Document Clustering**: Documents can be clustered hierarchically based on their content. This can be useful in organizing a large document collection, creating topic hierarchies, or identifying related documents.

4. **Image Segmentation**: In computer vision, hierarchical clustering can be used for image segmentation, where regions with similar colors or textures are grouped together. This can be applied in image processing and object recognition.

### Implementing Hierarchical Clustering in Python:

You can implement hierarchical clustering in Python using libraries like SciPy. Here's a high-level overview of the process:

1. **Data Preparation**: Prepare your data and ensure it is in a suitable format, typically a distance or similarity matrix.

2. **Hierarchical Clustering**: Use functions like `scipy.cluster.hierarchy.linkage` to perform hierarchical clustering based on your chosen linkage criterion.

3. **Dendrogram Visualization**: You can use `scipy.cluster.hierarchy.dendrogram` to create a dendrogram and visualize the clustering hierarchy.

4. **Cutting the Dendrogram**: Decide at what level of the dendrogram you want to cut it to obtain your clusters. This will give you a specific number of clusters based on your data and the linkage method you chose.

By exploring different levels of the dendrogram, you can gain insights into the hierarchical structure of your data and choose the level of granularity that best fits your analysis.

Hierarchical clustering is a versatile method for discovering patterns and structures in your data and can be a valuable tool in data exploration and analysis.



# Mathematical Methdologies


<b>Hierarchical clustering </b> (also called hierarchical cluster analysis or HCA) is a method of cluster analysis which seeks to build a hierarchy of clusters. Strategies for hierarchical clustering generally fall into two types:

- <b>Agglomerative </b>: This is a "bottom-up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
- <b>Divisive </b>: This is a "top-down" approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.

In general, the merges and splits are determined in a greedy manner. The results of hierarchical clustering are usually presented in a dendrogram.

## Agglomerative Clustering

Initially each data point is considered as an individual cluster. At each iteration, the similar clusters merge with other clusters until 1/ K clusters are formed.

The main advantage is that we don’t need to specify the number of clusters, this comes with a price: performance $O(n^3)$. In sklearn’s implementation, we can specify the number of clusters to assist the algorithm’s performance.

### Algorithm
- Compute the proximity matrix
- Let each data point be a cluster
- Repeat: Merge two closest clusters and update the proximity matrix until 1/ K cluster remains

Ex. - We have six data points {A,B,C,D,E,F}.

- In the initial step, we consider all the six data points as individual clusters as shown in the image below.



- The first step is to determine which elements to merge in a cluster. Usually, we want to take the two closest elements, according to the chosen distance.We construct a distance matrix at this stage, where the number in the i-th row j-th column is the distance between the i-th and j-th elements. Then, as clustering progresses, rows and columns are merged as the clusters are merged and the distances updated.

#### Computation of proximity/distance matrix

The choice of an appropriate metric will influence the shape of the clusters, as some elements may be close to one another according to one distance and farther away according to another. For example, in a 2-dimensional space, the distance between the point (1,0) and the origin (0,0) is always 1 according to the usual norms, but the distance between the point (1,1) and the origin (0,0) can be 2 under Manhattan distance, $\sqrt2$ under Euclidean distance, or 1 under maximum distance.

Some commonly used metrics for hierarchical clustering are:



For text or other non-numeric data, metrics such as the Hamming distance or Levenshtein distance are often used.

- Similar clusters are merged together and formed as a single cluster. Let’s consider B,C, and D,E are similar clusters that are merged in step two. Now, we’re left with four clusters which are A, BC, DE, F. To calculate the proximity between two clusters, we need to define the distance between them. Usually the distance is one of the following:
     - The maximum distance between elements of each cluster (also called  <b> complete-linkage clustering </b>)
     - The minimum distance between elements of each cluster (also called <b> single-linkage clustering </b>)
     - The mean distance between elements of each cluster (also called <b> average linkage clustering </b>)
     - The sum of all intra-cluster variance.
- Again calculate the proximity of new clusters and merge the similar clusters to form new clusters A, BC, DEF.
- Calculate the proximity of the new clusters. The clusters DEF and BC are similar and merged together to form a new cluster. We’re now left with two clusters A, BCDEF.
- Finally, all the clusters are merged together and form a single cluster.



The Hierarchical clustering Technique can be visualized using a Dendrogram.
A Dendrogram is a tree-like diagram that records the sequences of merges or splits.



