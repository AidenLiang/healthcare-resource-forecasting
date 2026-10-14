import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Load the dataset
data = pd.read_csv('forecasting_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Select the time series
time_series = data['ICU Beds in Use - Total']

# Check stationarity
result = adfuller(time_series.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# If not stationary, differencing the data
if result[1] > 0.05:
    time_series = time_series.diff().dropna()

# Plotting ACF
plt.figure(figsize=(14, 7))
plt.subplot(121)
plot_acf(time_series.dropna(), ax=plt.gca(), lags=40)
plt.title('Autocorrelation Function')

# Plotting PACF
plt.subplot(122)
plot_pacf(time_series.dropna(), ax=plt.gca(), lags=40, method='ywm')
plt.title('Partial Autocorrelation Function')

plt.tight_layout()
plt.show()
