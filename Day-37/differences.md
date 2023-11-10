# Key Concepts of Gradient Descent 


## **Batch Gradient Descent:**
```python
Copy code
def batch_gradient_descent(X, y, learning_rate, iterations):
    m = len(y)
    theta = np.zeros(X.shape[1])
    
    for iteration in range(iterations):
        gradient = (1/m) * np.dot(X.T, (np.dot(X, theta) - y))
        theta -= learning_rate * gradient
    
    return theta
```

### **Explanation:**

In Batch Gradient Descent, the entire dataset is used to calculate the gradient at each iteration.
It provides an accurate estimate of the gradient.
It can be computationally expensive for large datasets as it requires calculating the gradient for the entire dataset in each iteration.

## **Stochastic Gradient Descent (SGD):**



```python

def stochastic_gradient_descent(X, y, learning_rate, iterations):
    m = len(y)
    theta = np.zeros(X.shape[1])
    
    for iteration in range(iterations):
        for i in range(m):
            random_index = np.random.randint(0, m)
            xi = X[random_index]
            yi = y[random_index]
            gradient = 2 * xi.T.dot(xi.dot(theta) - yi)
            theta -= learning_rate * gradient
    
    return theta
```
### **Explanation:**

In Stochastic Gradient Descent, a single randomly chosen data point is used to estimate the gradient at each iteration.
It's computationally more efficient than Batch GD but can be noisy and lead to oscillations in the convergence path.
The random selection of data points introduces variability.

## **Mini-Batch Gradient Descent:**



```python

def mini_batch_gradient_descent(X, y, learning_rate, iterations, batch_size):
    m = len(y)
    theta = np.zeros(X.shape[1])
    
    for iteration in range(iterations):
        for i in range(0, m, batch_size):
            xi = X[i:i+batch_size]
            yi = y[i:i+batch_size]
            gradient = (1/batch_size) * xi.T.dot(xi.dot(theta) - yi)
            theta -= learning_rate * gradient
    
    return theta
```

### **Explanation:**

- In Mini-Batch Gradient Descent, a small random subset (mini-batch) of the data is used to estimate the gradient.
- It combines the advantages of both Batch GD and SGD.
- It is computationally efficient and provides a balance between accuracy and speed.
**

# **Key Differences:-** 
- Batch Gradient Descent uses the entire dataset in each iteration for gradient calculation, making it computationally expensive but accurate.
- Stochastic Gradient Descent (SGD) uses a single randomly chosen data point in each iteration, making it computationally efficient but noisy and less accurate.
- Mini-Batch Gradient Descent uses a small random subset of the data in each iteration, offering a balance between accuracy and efficiency.
- The choice of which variant to use depends on your specific problem and dataset size. Mini-Batch Gradient Descent is often a popular choice for many machine learning tasks.