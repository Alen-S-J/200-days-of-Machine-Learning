# IMPLEMENTATION OF BASIC NEURAL NETWORK

### Theoretical Aspect of a Basic Neural Network:

A basic neural network is composed of layers of interconnected artificial neurons or nodes. It typically consists of three main types of layers:

1. **Input Layer:** The input layer receives data or features and passes them to the network. Each neuron in the input layer corresponds to a feature in the input data. The values from this layer are directly passed to the next layer.

2. **Hidden Layer(s):** These layers are located between the input and output layers. Each neuron in a hidden layer receives weighted inputs from the previous layer, applies an activation function, and passes the result to the next layer. Hidden layers enable the network to learn complex patterns in the data.

3. **Output Layer:** The output layer produces the final predictions or results of the neural network. The number of neurons in this layer depends on the specific task (e.g., regression, classification), and the activation function used may vary accordingly.

#### Activation Functions:

Activation functions introduce non-linearity into the neural network, enabling it to model complex relationships in data. Common activation functions include:

- **Sigmoid:** Sigmoid activation function squashes the input into the range [0, 1]. It's commonly used in the output layer for binary classification problems.

- **ReLU (Rectified Linear Unit):** ReLU activation function is widely used in hidden layers. It returns the input if it's positive and zero otherwise. It helps mitigate the vanishing gradient problem.

- **Tanh (Hyperbolic Tangent):** Tanh activation function squashes the input into the range [-1, 1]. It's similar to the sigmoid but has a zero-centered output.

### Sample Code in Python with NumPy:

Here's a simple Python implementation of a feedforward neural network with one hidden layer using NumPy:

```python
import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize random weights and biases for the input and hidden layers
input_size = 2
hidden_size = 3
output_size = 1

weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))

# Initialize random weights and biases for the hidden and output layers
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.zeros((1, output_size))

# Define the forward pass function
def forward(X):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output = np.dot(hidden_output, weights_hidden_output) + bias_output
    return output

# Example input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Forward pass
output = forward(X)
print("Predictions:")
print(output)


This code represents a simple neural network with random weights. You would typically train the network by adjusting the weights and biases using backpropagation and a loss function, but this basic implementation shows the forward pass to get predictions.