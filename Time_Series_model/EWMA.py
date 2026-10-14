import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from metrics import mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error, mean_absolute_scaled_error


# Load the dataset
data = pd.read_csv('forecasting_data_cleaned.csv')

# Assuming 'data' is a pandas DataFrame and the time series is in a column named 'ICU Beds in Use - Total'
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Select the time series
time_series = data['ICU Beds in Use - Total']

# Split the data into train and test
train_size = int(len(time_series) * 0.8)
train, test = time_series[:train_size], time_series[train_size:]

# Applying EWMA model with a span of 7 days as an example
# The `span` argument corresponds to the window size for calculating the weighted average
ewma_model = train.ewm(span=7, adjust=False).mean()

# Forecast using the last value of the EWMA as a naive forecast
# This is a common approach for a basic EWMA model, though more sophisticated methods are available
forecast_series = pd.Series(ewma_model.iloc[-1], index=test.index)

# Calculate MAE
mae = mean_absolute_error(test, forecast_series)
print('MAE:', mae)

# Calculate MAPE
mape = mean_absolute_percentage_error(test, forecast_series)
print('MAPE:', mape)

# Calculate sMAPE
smape = symmetric_mean_absolute_percentage_error(test, forecast_series)
print('sMAPE:', smape)

# Calculate MASE
mase = mean_absolute_scaled_error(test, forecast_series, train)
print('MASE:', mase)




# Plot the results
plt.figure(figsize=(12, 7))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Test Data')
plt.plot(forecast_series.index, forecast_series, label='Forecast (EWMA)', color='red')
plt.title('EWMA Forecast')
plt.xlabel('Date')
plt.ylabel('ICU Beds in Use - Total')
plt.legend()
plt.show()
