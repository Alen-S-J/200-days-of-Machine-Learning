import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Function to generate sample time series data
def generate_time_series(start_date, end_date, freq='D', trend=0, seasonal_period=None, seasonal_strength=0, noise_level=0):
    index = pd.date_range(start=start_date, end=end_date, freq=freq)
    if seasonal_period:
        seasonal_component = seasonal_strength * np.sin(2 * np.pi * np.arange(len(index)) / seasonal_period)
    else:
        seasonal_component = 0
    trend_component = trend * np.arange(len(index))
    noise = np.random.normal(0, noise_level, size=len(index))
    return pd.Series(trend_component + seasonal_component + noise, index=index)

# Generate a sample time series dataset without seasonality
start_date = '2023-01-01'
end_date = '2023-12-31'
trend = 0.5
noise_level = 2
data_without_seasonality = generate_time_series(start_date, end_date, trend=trend, noise_level=noise_level)

# Generate a sample time series dataset with seasonality
seasonal_period = 30  # Assuming monthly seasonality
seasonal_strength = 5
data_with_seasonality = generate_time_series(start_date, end_date, seasonal_period=seasonal_period, seasonal_strength=seasonal_strength, trend=trend, noise_level=noise_level)

# Display the sample datasets
print("Sample Time Series Dataset without Seasonality:")
print(data_without_seasonality.head())

print("\nSample Time Series Dataset with Seasonality:")
print(data_with_seasonality.head())

# Plot the time series data
plt.figure(figsize=(10, 6))
plt.plot(series)
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Plot the ACF and PACF to determine the order of ARIMA(p, d, q)
plot_acf(series)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(series)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Define the ARIMA model parameters (p, d, q)
p = 1  # AR order
d = 1  # Differencing order
q = 1  # MA order

# Fit the ARIMA model
model = ARIMA(series, order=(p, d, q))
result = model.fit()

# Forecast future values
forecast_steps = 10  # Number of steps to forecast
forecast = result.forecast(steps=forecast_steps)

# Plot the original data and the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(series, label='Original Data')
plt.plot(np.arange(len(series), len(series) + forecast_steps), forecast, label='Forecast')
plt.title('ARIMA Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()