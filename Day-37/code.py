
import numpy as np

# Generate some sample data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# Initialize parameters
learning_rate = 0.01
iterations = 1000
m = len(X)
theta0 = 0
theta1 = 0

# Perform batch gradient descent
for iteration in range(iterations):
    # Calculate predictions
    predictions = theta0 + theta1 * X

    # Calculate gradients
    gradient_theta0 = (1/m) * np.sum(predictions - Y)
    gradient_theta1 = (1/m) * np.sum((predictions - Y) * X)

    # Update parameters
    theta0 -= learning_rate * gradient_theta0
    theta1 -= learning_rate * gradient_theta1

print("Optimal theta0:", theta0)
print("Optimal theta1:", theta1)



