import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np

# Load the cleaned dataset
data = pd.read_csv('forecasting_data_cleaned.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Define target and dynamically set all remaining columns as predictors
target = 'ICU Beds in Use - Total'
predictors = list(data.columns.drop(target))

# Split into features and target
X = data[predictors]
y = data[target]

# Step 1: Recursive Feature Elimination (RFE)
lr = LinearRegression()
rfe = RFE(estimator=lr, n_features_to_select=6)
rfe.fit(X, y)

# Step 2: LASSO CV
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lasso_cv = LassoCV(cv=5, max_iter=5000)
lasso_cv.fit(X_scaled, y)

# Identify selected features from RFE and LASSO
selected_features_rfe = [predictors[i] for i in range(len(predictors)) if rfe.support_[i]]
selected_features_lasso = [predictors[i] for i in range(len(predictors)) if lasso_cv.coef_[i] != 0]

# Union of selected features
selected_features = list(set(selected_features_rfe + selected_features_lasso))

print("Selected Features (RFE + LASSO):", selected_features)

# Create a new dataframe with the selected features and the target
selected_data = data[selected_features + [target]]

# Save the new dataset to a new CSV file
selected_data.to_csv('selected_healthcare_data.csv', index=True)

# Preview of the new dataset
print(selected_data.head())
