import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Sample data generation
np.random.seed(0)
n = 100
time = pd.date_range('2022-01-01', periods=n)
data = np.random.randn(n)
df = pd.DataFrame(data, index=time, columns=['Value'])

# ARIMA Model fitting
order = (1, 1, 1)  # ARIMA(p, d, q) order
model = ARIMA(df['Value'], order=order)
results = model.fit()

# Plot ACF and PACF to determine AR and MA components
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df['Value'], lags=20, ax=ax1)
plot_pacf(df['Value'], lags=20, ax=ax2)
plt.show()

# Print summary of the ARIMA model
print(results.summary())
