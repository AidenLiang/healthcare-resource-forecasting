import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from metrics import mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error, mean_absolute_scaled_error


# Load your cleaned dataset
data = pd.read_csv('forecasting_data_cleaned.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.index = pd.DatetimeIndex(data.index).to_period('D')

# Selecting a series to forecast
series = data['ICU Beds in Use - Total']

# Train/test split
split_point = int(len(series) * 0.8)
train, test = series[:split_point], series[split_point:]


# from pmdarima import auto_arima

# Running auto_arima
# model = auto_arima(train, seasonal=False, stepwise=True, trace=True)


# Model fitting
model = ARIMA(train, order=(2,1,4))
fitted_model = model.fit()

# Predictions using steps
steps = len(test)
predictions = fitted_model.get_forecast(steps=steps)
prediction_means = predictions.predicted_mean

# Set prediction index to test index if needed
prediction_means.index = test.index

# Plotting
plt.figure(figsize=(10,5))
plt.plot(train.index.to_timestamp(), train, label='Train')
plt.plot(test.index.to_timestamp(), test, label='Test')
plt.plot(prediction_means.index.to_timestamp(), prediction_means, label='Predicted', color='red')
plt.title('ICU Beds in Use Forecast')
plt.legend()
plt.show()

# Calculate RMSE, MAPE, sMAPE, MASE
rmse = np.sqrt(mean_squared_error(test, prediction_means))
mape = mean_absolute_percentage_error(test, prediction_means)
smape = symmetric_mean_absolute_percentage_error(test, prediction_means)
mase = mean_absolute_scaled_error(test, prediction_means, train.iloc[:-len(test)])

# Print out metrics
print('RMSE:', rmse)
print('MAPE:', mape)
print('sMAPE:', smape)
print('MASE:', mase)

# Print AIC, BIC, and R-squared values
print('AIC:', fitted_model.aic)
print('BIC:', fitted_model.bic)
