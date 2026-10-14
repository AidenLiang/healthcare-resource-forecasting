import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the cleaned dataset
data = pd.read_csv('forecasting_data_cleaned.csv')

# Convert date from string to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Create new features from the Date
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Optionally, drop the original Date column if it's no longer needed
data.drop('Date', axis=1, inplace=True)

# Select features and target
target = 'ICU Beds in Use - Total'
features = data.columns.drop(target)  # Assume all other columns are numeric and relevant

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=200)  # 100 trees in the forest

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Predict on the testing data
predictions = rf_model.predict(X_test)

# Calculate the RMSE
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f'RMSE: {rmse}')

# Feature importance
feature_importances = pd.Series(rf_model.feature_importances_, index=features)
print(feature_importances.sort_values(ascending=False))

# Plotting feature importances (Optional)
import matplotlib.pyplot as plt
feature_importances.sort_values().plot(kind='barh', figsize=(8,6))
plt.title('Feature Importance')
plt.show()

# Scatter plot to compare the actual and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Diagonal line
plt.xlabel('Actual ICU Beds in Use')
plt.ylabel('Predicted ICU Beds in Use')
plt.title('Actual vs. Predicted ICU Beds in Use (Random Forest)')
plt.show()

# Line plot for the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual')
plt.plot(predictions, label='Predicted', alpha=0.7)
plt.xlabel('Test Case Index')
plt.ylabel('ICU Beds in Use')
plt.title('Actual vs. Predicted ICU Beds in Use Over Test Set (Random Forest)')
plt.legend()
plt.show()