# Backpropagation and Optimization

### Mathematical Equations:

1. **Forward Propagation:**
   
   The forward propagation computes the predicted output of a neural network.
   
   - Input to a neuron (for a single example): $z = W \cdot x + b$
   - Activation function (e.g., sigmoid, ReLU): $a = \sigma(z)$
   
2. **Loss Function:**

   The loss function measures the error between the predicted output and the actual target.
   
   - For regression problems, Mean Squared Error (MSE) is commonly used: $L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$
   - For classification problems, Cross-Entropy is commonly used: $L(y, \hat{y}) = -\sum_i y_i \log(\hat{y}_i)$

3. **Backpropagation:**

   Backpropagation computes gradients of the loss with respect to the model parameters.

   - Gradient of the loss with respect to the neuron's output: $\frac{\partial L}{\partial a}$
   - Gradient of the loss with respect to the neuron's input: $\frac{\partial L}{\partial z}$
   - Gradient of the loss with respect to weights: $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial W} = x \cdot \frac{\partial L}{\partial z}$
   - Gradient of the loss with respect to bias: $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}$

4. **Gradient Descent:**

   Gradient Descent is an optimization algorithm to update model parameters.

   - Weight update rule: $W \leftarrow W - \alpha \frac{\partial L}{\partial W}$
   - Bias update rule: $b \leftarrow b - \alpha \frac{\partial L}{\partial b}$

### Sample Use Case Scenario with Markdown Code:

Let's consider a simple scenario of binary classification using a single neuron with the sigmoid activation function. We'll use Markdown to provide an example of code for forward propagation, loss calculation, backpropagation, and gradient descent.

```markdown
### Binary Classification Example

1. **Forward Propagation:**

   ```python
   # Inputs and weights
   x = 0.5
   W = 0.3
   b = 0.2

   # Neuron input
   z = W * x + b

   # Activation function (sigmoid)
   a = 1 / (1 + exp(-z))
   ```

2. **Loss Calculation:**

   Assuming the target label is `y = 1`:

   ```python
   # Mean Squared Error Loss
   L = 0.5 * (1 - a)**2
   ```

3. **Backpropagation:**

   ```python
   # Gradient of the loss with respect to the neuron's output
   dL_da = a - 1

   # Gradient of the neuron's input
   da_dz = a * (1 - a)

   # Gradient of the loss with respect to weights and bias
   dL_dW = x * dL_da * da_dz
   dL_db = dL_da * da_dz
   ```

4. **Gradient Descent:**

   Updating weights and bias using the learning rate $\alpha$:

   ```python
   learning_rate = 0.1
   W -= learning_rate * dL_dW
   b -= learning_rate * dL_db
```



This is a simplified example, but it demonstrates the core concepts of forward propagation, loss calculation, backpropagation, and gradient descent in a binary classification scenario. You can apply these concepts to more complex neural networks and real-world applications in machine learning.
