import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
data = pd.read_csv('forecasting_data_cleaned.csv')

# Convert date from string to datetime and create features
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data.drop('Date', axis=1, inplace=True)

# Define a threshold to categorize the ICU beds usage into binary classes
threshold = data['ICU Beds in Use - Total'].median()  # for example, using the median as the threshold
data['High_Utilization'] = (data['ICU Beds in Use - Total'] > threshold).astype(int)

# Select features and new binary target
target = 'High_Utilization'
features = data.columns.drop(['ICU Beds in Use - Total', target])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
lr_model = LogisticRegression(random_state=42)

# Train the model on the training data
lr_model.fit(X_train, y_train)

# Predict on the testing data
predictions = lr_model.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Plotting confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Logistic Regression')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()
