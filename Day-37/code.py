>>>>>>>>>>DOCIFY-START - bwzfjtlyyrzt >>>>>>>>>>
# This is the method that implements the gradient descent algorithm. We do not need to copy the code here because it relies on numpy
>>>>>>>>>>DOCIFY-END - bwzfjtlyyrzt >>>>>>>>>>

import numpy as np
# Generate some sample data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# Initialize parameters
learning_rate = 0.01
iterations = 1000
m = len(X)
θ0 = 0
θ1 = 0

# Perform batch gradient descent
for iteration in range(iterations):
    # Calculate predictions
    predictions = θ0 + θ1 * X

    # Calculate gradients
    gradient_θ0 = (1/m) * np.sum(predictions - Y)
    gradient_θ1 = (1/m) * np.sum((predictions - Y) * X)

    # Update parameters
    θ0 -= learning_rate * gradient_θ0
    θ1 -= learning_rate * gradient_θ1

print("Optimal θ0:", θ0)
print("Optimal θ1:", θ1)


