import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
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
features = data.columns.drop(target)

# Determine the split point for the last 20% of data
split_point = int(len(data) * 0.8)
X_train, X_test = data[features].iloc[:split_point], data[features].iloc[split_point:]
y_train, y_test = data[target].iloc[:split_point], data[target].iloc[split_point:]

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Support Vector Regression model
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_model.fit(X_train_scaled, y_train)

# Predict on the testing data
predictions = svr_model.predict(X_test_scaled)

# Calculate the RMSE
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f'SVR RMSE: {rmse}')

# Scatter plot to compare the actual and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual ICU Beds in Use')
plt.ylabel('Predicted ICU Beds in Use')
plt.title('Actual vs. Predicted ICU Beds in Use (SVR)')
plt.show()

# Line plot for the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual')
plt.plot(predictions, label='Predicted', alpha=0.7)
plt.xlabel('Test Case Index')
plt.ylabel('ICU Beds in Use')
plt.title('Actual vs. Predicted ICU Beds in Use Over Test Set (SVR)')
plt.legend()
plt.show()
