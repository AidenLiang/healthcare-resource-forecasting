import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
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


# Define the SARIMA model, replace order and seasonal_order with appropriate values
# For example, (1, 1, 1) for ARIMA parameters and (1, 1, 1, 7) for seasonal parameters
# with a seasonal period of 7 days, if we suspect weekly seasonality.
model = SARIMAX(train, 
                order=(0, 0, 0),               # Non-seasonal parameters
                seasonal_order=(0, 1, 2, 7))   # Seasonal parameters

# Fit the model
result = model.fit()

# Forecast
forecast = result.get_forecast(steps=len(test))
forecast_series = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()
#forecast_index = test.index


# Set prediction index to test index if needed
forecast_series.index = test.index

# Calculate RMSE, MAPE, sMAPE, MASE
rmse = np.sqrt(mean_squared_error(test, forecast_series))
mape = mean_absolute_percentage_error(test, forecast_series)
smape = symmetric_mean_absolute_percentage_error(test, forecast_series)
mase = mean_absolute_scaled_error(test, forecast_series, train.iloc[:-len(test)])

# Print out metrics
print('RMSE:', rmse)
print('MAPE:', mape)
print('sMAPE:', smape)
print('MASE:', mase)

# Print AIC, BIC, and R-squared values
print('AIC:', result.aic)
print('BIC:', result.bic)

# Calculate R-squared
# Note: This is an unconventional metric for time series and may not provide meaningful insight.
ss_res = np.sum(result.resid**2)
ss_tot = np.sum((train - np.mean(train))**2)
r_squared = 1 - ss_res / ss_tot
print('R-squared:', r_squared)

# Plot the results
plt.figure(figsize=(12, 7))
plt.plot(train.index.to_timestamp(), train, label='Training Data')  # Convert Period to Timestamp
plt.plot(test.index.to_timestamp(), test, label='Test Data')        # Convert Period to Timestamp
plt.plot(forecast_series.index.to_timestamp(), forecast_series, label='Forecast')  # Convert Period to Timestamp
plt.fill_between(forecast_series.index.to_timestamp(), forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', label='Confidence Interval')  # Convert Period to Timestamp
plt.title('SARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('ICU Beds in Use - Total')
plt.legend()
plt.show()

