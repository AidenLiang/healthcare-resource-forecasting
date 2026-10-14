import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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

# Determine the split point for the last 20% of data
split_point = int(len(data) * 0.8)
X_train, X_test = data[features].iloc[:split_point], data[features].iloc[split_point:]
y_train, y_test = data[target].iloc[:split_point], data[target].iloc[split_point:]

# Initialize the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)  # One tree

# Train the model on the training data
dt_model.fit(X_train, y_train)

# Predict on the testing data
predictions = dt_model.predict(X_test)

# Calculate the RMSE
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f'RMSE: {rmse}')

# Feature importance for Decision Tree
feature_importances = pd.Series(dt_model.feature_importances_, index=features)
print(feature_importances.sort_values(ascending=False))

# Plotting feature importances for Decision Tree (Optional)
plt.figure(figsize=(8,6))
feature_importances.sort_values().plot(kind='barh')
plt.title('Feature Importance (Decision Tree)')
plt.show()

# Scatter plot to compare the actual and predicted values for Decision Tree
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Diagonal line
plt.xlabel('Actual ICU Beds in Use')
plt.ylabel('Predicted ICU Beds in Use')
plt.title('Actual vs. Predicted ICU Beds in Use (Decision Tree)')
plt.show()

# Line plot for the actual and predicted values for Decision Tree
plt.figure(figsize=(10, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual')
plt.plot(predictions, label='Predicted', alpha=0.7)
plt.xlabel('Test Case Index')
plt.ylabel('ICU Beds in Use')
plt.title('Actual vs. Predicted ICU Beds in Use Over Test Set (Decision Tree)')
plt.legend()
plt.show()
