import pandas as pd
from sklearn.metrics import mean_squared_error

# Load the cleaned dataset
data = pd.read_csv('../enhanced_healthcare_weather_data.csv')

# Assuming 'ICU Beds in Use - Total' is sorted chronologically
actual_values = data['ICU Beds in Use - Total']

# Create the naive forecast
# The naive forecast "predicts" each value to be the same as the previous value
naive_forecast = actual_values.shift(1)

# Drop the first na value that results from the shift operation
actual_values = actual_values[1:]
naive_forecast = naive_forecast[1:]

# Calculate the RMSE for the naive forecast
rmse = mean_squared_error(actual_values, naive_forecast, squared=False)
print(f'Naive Forecast RMSE: {rmse}')

# You can plot the actual vs. naive forecast for visualization (optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(actual_values.index, actual_values, label='Actual')
plt.plot(actual_values.index, naive_forecast, label='Naive Forecast', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('ICU Beds in Use')
plt.title('Actual vs Naive Forecast')
plt.legend()
plt.show()
