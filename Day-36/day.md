# Backpropagation Basics

#### 1. Understanding Basic Principles:
- Backpropagation is a supervised learning algorithm used to train neural networks.
- It involves a forward pass (calculating predictions) and a backward pass (updating weights based on errors).
- The core idea is to minimize the difference between predicted and actual outputs.

#### 2. Chain Rule in Neural Networks:
- In the context of neural networks, the chain rule is crucial for calculating gradients.
- If you have a function $$ F(x) $$ composed of multiple functions like $$ F(x) = g(h(x)) $$, the chain rule states that $$ $$frac{dF}{dx} = $$frac{dg}{dh} $$cdot $$frac{dh}{dx} $$.
- In neural networks, this helps us compute the gradient of the loss with respect to the weights.

#### 3. Backpropagation for Weight Update:
- The weight update in backpropagation can be expressed as: 
  $$ $$text{New Weight} = $$text{Old Weight} - $$text{Learning Rate} $$times $$frac{$$partial $$text{Loss}}{$$partial $$text{Weight}} $$
- This formula ensures that we move the weights in the direction that reduces the loss.

#  Mathematics of Backpropagation

#### 1. Mathematical Expressions:
- The forward pass involves calculating the weighted sum of inputs and applying an activation function: 
  $$ z = $$sum_i (w_i $$cdot x_i) + b $$
  $$ a = $$text{activation}(z) $$
- The loss function measures the difference between predicted and actual outputs.

#### 2. Activation Function and Derivatives:
- Common activation functions include sigmoid, tanh, and ReLU.
- The derivative of the activation function is crucial for the chain rule in backpropagation.
- Example: Derivative of sigmoid activation $$ $$frac{d}{dz}($$sigma(z)) = $$sigma(z) $$cdot (1 - $$sigma(z)) $$.

#### 3. Code Example (in Python with NumPy):
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Forward pass
inputs = np.array([1.0, 2.0, 3.0])
weights = np.array([0.2, 0.5, -0.1])
bias = 0.1

z = np.dot(inputs, weights) + bias
activation = sigmoid(z)

# Backward pass
target = 1.0  # Actual output
loss = 0.5 * (target - activation)**2  # Mean Squared Error

# Calculate gradient using chain rule
d_loss_d_activation = -(target - activation)
d_activation_dz = sigmoid_derivative(z)
dz_dw = inputs

# Chain rule to get gradient of loss with respect to weights
gradient = d_loss_d_activation * d_activation_dz * dz_dw

# Update weights
learning_rate = 0.01
weights -= learning_rate * gradient
