import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Load the cleaned dataset
data = pd.read_csv('../enhanced_healthcare_weather_data.csv')

# Convert date from string to datetime and create new features
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data.drop('Date', axis=1, inplace=True)  # Remove the original Date column

# Define the target and features
target = 'ICU Beds in Use - Total'
features = data.columns.drop(target)  # All other columns are considered features

# Determine the split point for the last 20% of data
split_point = int(len(data) * 0.8)
X_train, X_test = data[features].iloc[:split_point], data[features].iloc[split_point:]
y_train, y_test = data[target].iloc[:split_point], data[target].iloc[split_point:]

# Create polynomial features
poly_degree = 2  # Degree of the polynomial features
poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Initialize and train the Polynomial Regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict on the testing data with polynomial features
predictions = model.predict(X_test_poly)

# Calculate the RMSE for the predictions
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f'Polynomial Regression RMSE: {rmse}')

# Scatter plot to compare the actual and predicted values
plt.figure(figsize=(10,5))
plt.scatter(y_test, predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual ICU Beds in Use')
plt.ylabel('Predicted ICU Beds in Use')
plt.title('Actual vs Predicted ICU Beds in Use (Polynomial Regression)')
plt.show()

# Line plot for the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual')
plt.plot(predictions, label='Predicted', alpha=0.7)
plt.xlabel('Test Case Index')
plt.ylabel('ICU Beds in Use')
plt.title('Actual vs. Predicted ICU Beds in Use Over Test Set (Polynomial Regression)')
plt.legend()
plt.show()
