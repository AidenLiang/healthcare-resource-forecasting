# Recursive Feature Elimination
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# Load the cleaned dataset
data = pd.read_csv('../Test_dataset/forecasting_data_cleaned.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Define target and predictors
target = 'ICU Beds in Use - Total'
predictors = [
    'Month', 'TEMP', 'PRCP', 'HMDT', 'WND_SPD', 'ATM_PRESS'
]  # Example features, can add more based on dataset

# Split into features and target
X = data[predictors]
y = data[target]

# Create a linear regression model
lr = LinearRegression()

# Use RFE to rank features
rfe = RFE(estimator=lr, n_features_to_select=3)  # Select top 3 features, adjust as needed
rfe.fit(X, y)

# Print the ranking of each feature
print("Feature Ranking:")
for i, feature in enumerate(predictors):
    print(f"{feature}: {rfe.ranking_[i]}")

# Get the selected features
selected_features = [predictors[i] for i in range(len(predictors)) if rfe.support_[i]]
print("Selected Features:", selected_features)
