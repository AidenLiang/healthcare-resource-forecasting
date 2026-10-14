import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import numpy as np
from metrics import mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error, mean_absolute_scaled_error

# Load the dataset
data = pd.read_csv('forecasting_data_cleaned.csv')

# Assuming 'data' is a pandas DataFrame and the time series is in a column named 'ICU Beds in Use - Total'
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Convert index to period if necessary, 'D' for daily frequency
data.index = data.index.to_period('D')

# Select the time series
time_series = data['ICU Beds in Use - Total']

# Split the data into train and test
train_size = int(len(time_series) * 0.8)
train, test = time_series[:train_size], time_series[train_size:]

# Define the Holt-Winters model
# Replace 'seasonal_periods' with an appropriate value to reflect seasonality
seasonal_periods = 7  # For example, 7 days if we suspect weekly seasonality
model = ExponentialSmoothing(
    train, trend='add', seasonal='add', seasonal_periods=seasonal_periods
)

# Fit the model
result = model.fit()

# Forecast
forecast_series = result.forecast(steps=len(test))

# Calculate confidence intervals (not inherently available with Holt-Winters)
# You can compute approximate confidence intervals if needed.

# Calculate metrics
rmse = np.sqrt(mean_squared_error(test, forecast_series))
mape = mean_absolute_percentage_error(test, forecast_series)
smape = symmetric_mean_absolute_percentage_error(test, forecast_series)
mase = mean_absolute_scaled_error(test, forecast_series, train)

# Print out metrics
print('RMSE:', rmse)
print('MAPE:', mape)
print('sMAPE:', smape)
print('MASE:', mase)

# Plot the results
plt.figure(figsize=(12, 7))
plt.plot(train.index.to_timestamp(), train, label='Training Data')
plt.plot(test.index.to_timestamp(), test, label='Test Data')
plt.plot(forecast_series.index.to_timestamp(), forecast_series, label='Forecast')
plt.title('Holt-Winters Forecast')
plt.xlabel('Date')
plt.ylabel('ICU Beds in Use - Total')
plt.legend()
plt.show()
