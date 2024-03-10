
import numpy as np
import pandas as pd
from scipy import stats

# Generate sample time series data with missing values and outliers
np.random.seed(0)
date_range = pd.date_range(start='2024-01-01', end='2024-01-31')
ts_data = pd.Series(np.random.randn(len(date_range)), index=date_range)
ts_data[3:6] = np.nan  # introduce missing values
ts_data[20] = 5  # introduce an outlier

# Handling Missing Values
ts_data_filled = ts_data.fillna(method='ffill')  # Forward fill missing values
print("Time series data after filling missing values:\n", ts_data_filled)

# Handling Outliers
z_scores = np.abs(stats.zscore(ts_data_filled))
threshold = 3
outlier_indices = np.where(z_scores > threshold)[0]
ts_data_no_outliers = ts_data_filled.copy()
ts_data_no_outliers.iloc[outlier_indices] = np.nan  # Replace outliers with NaNs
ts_data_no_outliers = ts_data_no_outliers.interpolate()  # Interpolate to fill NaNs
print("\nTime series data after removing outliers:\n", ts_data_no_outliers)

# Handling Non-Stationarity (Detrending)
detrended_data = ts_data_no_outliers.diff().fillna(method='bfill')  # Differencing to remove trend
print("\nTime series data after detrending:\n", detrended_data)

# Modeling Approaches (Simple Moving Average)
rolling_mean = ts_data_no_outliers.rolling(window=7).mean()  # Calculate 7-day moving average
print("\n7-day moving average:\n", rolling_mean)


