
import pandas as pd
from sklearn.cluster import KMeans

# Load customer data
data = pd.read_csv("E:/200 days of Machine Learning/Day-21/data/Mall_Customers.csv")

# Feature selection (e.g., total purchase amount and frequency of purchases)
X = data[["Annual_Income_(k$)","Spending_Score"]]

# Choose the number of clusters (K) based on business knowledge or techniques like the Elbow method
k = 3

# Initialize and fit the K-means model
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# Add cluster labels to the dataset
data["Cluster"] = kmeans.labels_

# Visualize the clusters
import matplotlib.pyplot as plt

for cluster in range(k):
    plt.scatter(data[data["Cluster"] == cluster]["Annual_Income_(k$)"],
                data[data["Cluster"] == cluster]["Spending_Score"],
                label=f"Cluster {cluster}")

plt.xlabel("Annual_Income_(k$)")
plt.ylabel("Spending_Score")
plt.legend()
plt.show()
