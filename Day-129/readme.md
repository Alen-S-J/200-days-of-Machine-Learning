# **Day 129: Implementation of ARIMA and SARIMA Models**

1. **Install Required Libraries**: Make sure you have `pandas`, `numpy`, and `statsmodels` installed. You can install them using pip if you haven't already:

```bash
pip install pandas numpy statsmodels
```

2. **Load the Dataset**: Load your time series dataset into a pandas DataFrame.

3. **Preprocess the Data**: If necessary, preprocess the data by handling missing values, converting date-time columns, and ensuring the data is stationary (if not, you may need to difference it).

4. **Split the Data**: Split the data into training and testing sets.

5. **Build and Fit the Model**: Build an ARIMA or SARIMA model using the `ARIMA` or `SARIMAX` class from `statsmodels`. You'll need to determine the appropriate order parameters (p, d, q) for ARIMA and (p, d, q, s) for SARIMA based on the dataset.

6. **Evaluate the Model**: Evaluate the performance of the model using appropriate metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and visualize the actual vs. predicted values.

Here's a sample code demonstrating how to implement ARIMA and SARIMA models:

```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Step 1: Generate Custom Time Series Data
np.random.seed(42)
t = np.arange(100)
seasonality = 10 * np.sin(2 * np.pi * t / 12)  # Monthly seasonality
trend = 0.5 * t  # Linear trend
noise = np.random.normal(0, 2, size=len(t))  # Random noise
data = trend + seasonality + noise
date_range = pd.date_range(start='2020-01-01', periods=len(t), freq='M')
df = pd.DataFrame({'date': date_range, 'value': data})

# Step 2: Visualize the Data
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['value'], label='Original Data')
plt.title('Generated Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Step 3: Preprocess the Data (if needed)

# Step 4: Split the Data
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Step 5: Build and Fit the Models
# ARIMA Model
order_arima = (1, 1, 1)  # Example order parameters
model_arima = ARIMA(train['value'], order=order_arima)
model_arima_fit = model_arima.fit()

# SARIMA Model
order_sarima = (1, 1, 1)  # Example order parameters
seasonal_order_sarima = (1, 1, 1, 12)  # Example seasonal order parameters
model_sarima = SARIMAX(train['value'], order=order_sarima, seasonal_order=seasonal_order_sarima)
model_sarima_fit = model_sarima.fit()

# Step 6: Forecast
forecast_arima = model_arima_fit.forecast(steps=len(test))
forecast_sarima = model_sarima_fit.forecast(steps=len(test))

# Step 7: Evaluate the Models
mse_arima = mean_squared_error(test['value'], forecast_arima)
mse_sarima = mean_squared_error(test['value'], forecast_sarima)

print(f"MSE for ARIMA: {mse_arima}")
print(f"MSE for SARIMA: {mse_sarima}")

# Step 8: Visualize the Results
plt.figure(figsize=(10, 6))
plt.plot(test['date'], test['value'], label='Actual')
plt.plot(test['date'], forecast_arima, label='ARIMA Forecast')
plt.plot(test['date'], forecast_sarima, label='SARIMA Forecast')
plt.title('ARIMA and SARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

```

Remember to replace `'your_dataset.csv'` with the path to your dataset file. Also, adjust the order parameters (`order` and `seasonal_order`) according to your dataset characteristics. Finally, visualize the actual vs. predicted values to assess the model's performance visually.