```python
import numpy as np
from scipy.stats import bernoulli

# Define probability of success (e.g., getting a '1')
p = 0.3

# Generate a random sample following Bernoulli distribution
sample = bernoulli.rvs(p, size=10)

print("Bernoulli Distribution Sample:", sample)

``````


``````python

Output 

Bernoulli Distribution Sample: [0 0 1 0 1 0 0 1 0 0]

``````