
# Optimization Algorithms

### Basic Gradient Descent

Basic Gradient Descent minimizes the cost (or loss) function by iteratively updating model parameters based on the gradient of the cost function with respect to the parameters. The update rule is as follows:



```python
# Implementation of Gradient Descent
learning_rate = 0.01
num_iterations = 1000

for i in range(num_iterations):
    gradient = compute_gradient(cost_function, parameters)
    parameters -= learning_rate * gradient
```

### Momentum-based Optimization

Momentum-based optimization methods introduce a momentum term to accelerate convergence by giving the gradient update a "push" in the direction it was previously moving.

#### Momentum


```python
# Implementation of Momentum
learning_rate = 0.01
momentum = 0.9
velocity = 0  # Initial velocity

for i in range(num_iterations):
    gradient = compute_gradient(cost_function, parameters)
    velocity = momentum * velocity - learning_rate * gradient
    parameters += velocity
```

#### Nesterov Accelerated Gradient (NAG)

Nesterov Accelerated Gradient, often referred to as NAG, first makes a "predictive step" to approximate where the parameters will be in the next iteration, and then computes the gradient at that predicted position.


```python
# Implementation of Nesterov Accelerated Gradient (NAG)
learning_rate = 0.01
momentum = 0.9
velocity = 0  # Initial velocity

for i in range(num_iterations):
    parameters += momentum * velocity  # Predictive step
    gradient = compute_gradient(cost_function, parameters)
    velocity = momentum * velocity - learning_rate * gradient
    parameters += velocity
```

### Adaptive Learning Rate Methods

Adaptive learning rate methods adjust the learning rate during training based on the historical behavior of gradients.

#### RMSprop

RMSprop adapts the learning rate by scaling it with the moving average of the squared gradients.



```python
# Implementation of RMSprop
learning_rate = 0.01
decay_rate = 0.9
epsilon = 1e-7  # Small constant to prevent division by zero
moving_avg_sq = 0  # Initial moving average of squared gradients

for i in range(num_iterations):
    gradient = compute_gradient(cost_function, parameters)
    moving_avg_sq = decay_rate * moving_avg_sq + (1 - decay_rate) * gradient**2
    parameters -= (learning_rate / (sqrt(moving_avg_sq) + epsilon)) * gradient
```

#### Adam

Adam combines the concepts of momentum and RMSprop to adapt the learning rate and introduce bias correction for the moving averages.



```python
# Implementation of Adam
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7
m = 0  # Initialize 1st moment estimate
v = 0  # Initialize 2nd moment estimate
t = 0  # Initialize time step

for i in range(num_iterations):
    t += 1
    gradient = compute_gradient(cost_function, parameters)
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    parameters -= (learning_rate / (sqrt(v_hat) + epsilon)) * m_hat
```

By implementing and experimenting with these optimization algorithms, you can gain a better understanding of their behavior and how to choose appropriate hyperparameters for your specific machine learning tasks.