# Autoregressive (AR) and Moving Average (MA) components


Sample Python code using the `statsmodels` library to demonstrate how to fit an ARIMA model, and a brief explanation of the autoregressive (AR) and moving average (MA) components:

```python
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
```

**Theory:**

1. **Autoregressive (AR) Component:**
   - The autoregressive component (AR) of an ARIMA model represents the correlation between an observation and a certain number of lagged observations (i.e., observations from previous time steps).
   - In the context of the ARIMA(p, d, q) model, the AR component (p) is the number of lagged observations included in the model.
   - In the sample code, the AR component is set to 1 (`order = (1, 1, 1)`), meaning that the model includes one lagged observation.

2. **Moving Average (MA) Component:**
   - The moving average component (MA) of an ARIMA model represents the correlation between an observation and a residual error from a moving average model applied to lagged observations.
   - In the ARIMA(p, d, q) model, the MA component (q) is the number of lagged forecast errors included in the model.
   - In the sample code, the MA component is also set to 1 (`order = (1, 1, 1)`), indicating that the model includes one lagged forecast error.

By fitting the ARIMA model and examining the summary, you can see the coefficients and significance levels of the AR and MA components, which help in understanding the relationships between the current observation and its lagged values and forecast errors. Additionally, plotting the autocorrelation function (ACF) and partial autocorrelation function (PACF) aids in determining suitable values for the AR and MA components.