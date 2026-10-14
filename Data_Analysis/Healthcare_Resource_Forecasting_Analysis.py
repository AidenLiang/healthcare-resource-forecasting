import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Read the CSV file
file_name = 'COVID-19_Hospital_Capacity_Metrics_-_Historical.csv'
data = pd.read_csv(file_name)

# Convert 'Date' to a datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Set the date as the index of the dataframe
data.set_index('Date', inplace=True)

# 1. Capacity Planning

# Calculate the current trend of ventilator usage
ventilator_usage = data['Ventilators in Use - Total'].resample('W').mean()  # Resample data weekly
ventilator_usage.plot(title='Weekly Average Ventilator Usage')

# Fit a linear regression model to predict future ventilator needs
X = np.arange(len(ventilator_usage)).reshape(-1, 1)  # Time as independent variable
y = ventilator_usage.values  # Ventilator usage as dependent variable
model = LinearRegression().fit(X, y)

# Predict usage for the next 4 weeks
future_weeks = np.arange(len(ventilator_usage), len(ventilator_usage) + 4).reshape(-1, 1)
future_predictions = model.predict(future_weeks)

# Correcting the way to calculate predicted_dates
# Assuming the data is weekly, add 1 week to the last date in the index
start_date = ventilator_usage.index[-1] + pd.Timedelta(weeks=1)

# Generate predicted dates starting from the next week
predicted_dates = pd.date_range(start=start_date, periods=4, freq='W')

# Plotting the results
plt.figure(figsize=(10, 6))
ventilator_usage.plot(title='Weekly Average Ventilator Usage vs. Predicted Usage', label='Actual Usage')
plt.plot(predicted_dates, future_predictions, label='Predicted Usage', linestyle='--', marker='o', color='red')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Ventilator Usage')
plt.grid(True)
plt.show()



# 2. Utilization Trends

# Plotting the trend of ICU bed utilization
icu_usage = data['ICU Beds in Use - Total'].resample('W').mean()
icu_usage.plot(title='Weekly Average ICU Bed Usage')
plt.show()

# Futrue work: We can also add more plots for ventilator usage to compare trends


# 3. Forecasting Demand
# ARIMA example for ICU bed forecasting
from statsmodels.tsa.arima.model import ARIMA

# Create an ARIMA model for ICU Beds usage
icu_model = ARIMA(icu_usage, order=(1, 1, 1))  # Example order, should be optimized
icu_model_fit = icu_model.fit()

# Forecast the next 4 weeks
icu_forecast = icu_model_fit.forecast(steps=4)
print(icu_forecast)

# 4. Resource Allocation
# Compare the available resources to usage to determine potential shortages
data['Ventilators Shortage'] = data['Ventilators - Total Capacity'] - data['Ventilators in Use - Total']
data['ICU Beds Shortage'] = data['ICU Beds - Total Capacity'] - data['ICU Beds in Use - Total']

# Identify dates and locations with the most significant shortages
shortages = data.loc[data['Ventilators Shortage'] < 0, ['Ventilators Shortage', 'ICU Beds Shortage']]
print(shortages)


#5. Impact Analysis

# Correlation between lockdown periods and ICU bed usage
# Assuming 'lockdown' is a binary column indicating the presence of a lockdown

# Example of setting a hypothetical lockdown period
lockdown_start_date = '2020-03-15'
lockdown_end_date = '2020-05-15'

# Convert index back to a column if it's not already a datetime column
data.reset_index(inplace=True)

# Convert 'Date' to a datetime type if it hasn't been converted yet
data['Date'] = pd.to_datetime(data['Date'])

# Create a lockdown dummy variable based on the hypothetical dates
data['lockdown'] = (pd.to_datetime(lockdown_start_date) <= data['Date']) & (data['Date'] <= pd.to_datetime(lockdown_end_date))

# Assuming 'ICU Beds in Use - Total' is a column in your data, plot ICU bed usage during and outside lockdown periods
data.set_index('Date', inplace=True)  # Resetting Date as the index if it's not
data[data['lockdown']]['ICU Beds in Use - Total'].plot(style='r', label='During Lockdown')
data[~data['lockdown']]['ICU Beds in Use - Total'].plot(style='b', label='Outside Lockdown')
plt.legend()
plt.show()

# Futrue work: We would need lockdown start and end date columns for this analysis.
