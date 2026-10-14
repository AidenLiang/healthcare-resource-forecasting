import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load the cleaned dataset
data = pd.read_csv('forecasting_data_cleaned.csv')

# Convert date from string to datetime and create the 'Month' feature
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month

# Define the target and features (using only 'Month' as a feature)
target = 'ICU Beds in Use - Total'
features = ['Month']  # Only one feature for simple linear regression

# Determine the split point for the last 20% of data
split_point = int(len(data) * 0.8)
X_train, X_test = data[features].iloc[:split_point], data[features].iloc[split_point:]
y_train, y_test = data[target].iloc[:split_point], data[target].iloc[split_point:]

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the testing data
predictions = model.predict(X_test)

# Calculate the RMSE for the predictions
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f'Simple Linear Regression RMSE: {rmse}')

# Plotting actual vs predicted (Optional)
plt.figure(figsize=(10,5))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.5)
plt.scatter(X_test, predictions, color='red', label='Predicted', alpha=0.5)
plt.xlabel('Month')
plt.ylabel('ICU Beds in Use')
plt.title('Actual vs Predicted ICU Beds in Use by Month')
plt.legend()
plt.show()

# Optional: Plotting regression line over the data points
plt.figure(figsize=(10,5))
plt.scatter(X_test, y_test, color='blue', alpha=0.5)
plt.plot(X_test, predictions, color='red', linewidth=2)
plt.xlabel('Month')
plt.ylabel('ICU Beds in Use')
plt.title('Regression Line: ICU Beds in Use by Month')
plt.show()
