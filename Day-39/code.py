import numpy as np

# Define a simple neural network architecture
input_size = 2
hidden_size = 3
output_size = 1

# Initialize weights and biases
np.random.seed(0)
W1 = np.random.rand(hidden_size, input_size)
B1 = np.zeros((hidden_size, 1))
W2 = np.random.rand(output_size, hidden_size)
B2 = np.zeros((output_size, 1))

# Define the sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define your input data X and target values Y
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # Example input data
Y = np.array([[0, 1, 1, 0]])  # Example target values

# Forward pass
def forward_pass(X, W1, B1, W2, B2):
    Z1 = np.dot(W1, X) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + B2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

# Loss function (MSE)
def mean_squared_error(A2, Y):
    m = Y.shape[1]
    return np.sum((A2 - Y)**2) / (2*m)

# Backpropagation
def backward_pass(X, Y, Z1, A1, Z2, A2, W1, W2):
    m = Y.shape[1]
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    dB2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    dB1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, dB1, dW2, dB2

# Gradient Descent update
def gradient_descent(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate):
    W1 -= learning_rate * dW1
    B1 -= learning_rate * dB1
    W2 -= learning_rate * dW2
    B2 -= learning_rate * dB2

# Training loop
learning_rate = 0.1
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    Z1, A1, Z2, A2 = forward_pass(X, W1, B1, W2, B2)
    
    # Compute and print the loss
    loss = mean_squared_error(A2, Y)
    print(f"Epoch {epoch}: Loss = {loss}")
    
    # Backpropagation
    dW1, dB1, dW2, dB2 = backward_pass(X, Y, Z1, A1, Z2, A2, W1, W2)
    
    # Update parameters
    gradient_descent(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate)

# After training, you can use the trained network for prediction.

def predict(X, W1, B1, W2, B2):
    Z1, A1, Z2, A2 = forward_pass(X, W1, B1, W2, B2)
    return A2  # A2 contains the network's output

# Example input for prediction
new_input = np.array([[0.2, 0.3], [0.7, 0.8]]).T

# Perform prediction
predictions = predict(new_input, W1, B1, W2, B2)

print("Predictions:")
print(predictions)