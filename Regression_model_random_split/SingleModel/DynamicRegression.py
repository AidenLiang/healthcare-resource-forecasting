import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the cleaned dataset
data = pd.read_csv('forecasting_data_cleaned.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Create a 'Month' column from 'Date' if not already created
data['Month'] = data.index.month

# Define target and predictors
target = 'ICU Beds in Use - Total'
predictors = ['Month']  # Example: Using only Month as external regressor

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Build the model
model = SARIMAX(train_data[target], exog=train_data[predictors], order=(1,1,1), seasonal_order=(0,0,0,0))

# Fit the model
results = model.fit(disp=False)

# Predict on the testing data
start = len(train_data)
end = len(data) - 1
predictions = results.predict(start=start, end=end, exog=test_data[predictors])

# Calculate RMSE
rmse = mean_squared_error(test_data[target], predictions, squared=False)
print(f'Dynamic Regression RMSE: {rmse}')

# Plotting actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data[target], label='Training Data')
plt.plot(test_data.index, test_data[target], label='Actual Test Data')
plt.plot(test_data.index, predictions, label='Predicted Data', color='red')
plt.title('Dynamic Regression Forecast vs Actuals')
plt.legend()
plt.show()