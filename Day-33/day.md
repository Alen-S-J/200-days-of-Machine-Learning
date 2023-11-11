# Activation function

#### Sigmoid Activation Function:
The sigmoid function squashes input values to the range [0, 1]. It's often used in the output layer for binary classification problems, as it maps the output to a probability.

**Mathematical Expression:**
$$
$$sigma(x) = $$frac{1}{1 + e^{-x}}
$$

**Code Sample in Python:**
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Example usage:
x = 2.0
result = sigmoid(x)
print(result)
```

**Use Case Scenario:**
- Binary classification problems, where you want to predict probabilities (e.g., spam detection).
- It was more popular in the past but has been largely replaced by ReLU in hidden layers due to some disadvantages.

#### Tanh (Hyperbolic Tangent) Activation Function:
The tanh function squashes input values to the range [-1, 1]. It is zero-centered, which helps mitigate the vanishing gradient problem.

**Mathematical Expression:**
$$
$$tanh(x) = $$frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

**Code Sample in Python:**
```python
import numpy as np

def tanh(x):
    return np.tanh(x)

# Example usage:
x = 2.0
result = tanh(x)
print(result)
```

**Use Case Scenario:**
- Hidden layers of a neural network, especially when data has zero-mean (e.g., text data).
- It helps address vanishing gradients better than the sigmoid function.

#### ReLU (Rectified Linear Unit) Activation Function:
ReLU is a piecewise linear function that returns the input for positive values and zero for negative values. It's the most popular activation function for hidden layers due to its simplicity and effectiveness.

**Mathematical Expression:**
$$
f(x) = $$max(0, x)
$$

**Code Sample in Python:**
```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

# Example usage:
x = 2.0
result = relu(x)
print(result)
```

**Use Case Scenario:**
- Hidden layers of deep neural networks, especially in deep learning models (e.g., image classification, natural language processing).
- It's computationally efficient and helps mitigate the vanishing gradient problem.

The choice of activation function depends on the problem you're trying to solve and the characteristics of your data. Sigmoid and tanh are useful in specific cases, but ReLU has become the default choice for most applications due to its training efficiency and better performance in deep networks.