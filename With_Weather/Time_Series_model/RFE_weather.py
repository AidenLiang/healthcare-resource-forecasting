# Recursive Feature Elimination
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import numpy as np

# Load the cleaned dataset
data = pd.read_csv('../enhanced_healthcare_weather_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Define target and dynamically set all remaining columns as predictors
target = 'ICU Beds in Use - Total'
predictors = list(data.columns.drop(target))

# Split into features and target
X = data[predictors]
y = data[target]

# Create a linear regression model
lr = LinearRegression()

# Use RFE to rank features
rfe = RFE(estimator=lr, n_features_to_select=10)  # Select the top 5 features, adjust as needed
rfe.fit(X, y)

# Print the ranking of each feature
print("Feature Ranking:")
for i, feature in enumerate(predictors):
    print(f"{feature}: {rfe.ranking_[i]}")

# Get the selected features
selected_features = [predictors[i] for i in range(len(predictors)) if rfe.support_[i]]
print("Selected Features:", selected_features)

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create LASSO model with cross-validation
lasso = LassoCV(cv=5, max_iter=100000)
lasso.fit(X_scaled, y)

# Print coefficients
print("Feature Coefficients:")
for i, feature in enumerate(predictors):
    print(f"{feature}: {lasso.coef_[i]}")

# Get selected features
selected_lasso_features = [predictors[i] for i in range(len(predictors)) if lasso.coef_[i] != 0]
print("Selected LASSO Features:", selected_lasso_features)

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

# Create a grid of possible alpha values
alphas = np.logspace(-3, 2, 10)  # Example: 10 values from 0.001 to 100

# Create Lasso model
lasso = Lasso(max_iter=100000)  # Increase iterations

# GridSearchCV for finding the best alpha
grid_search = GridSearchCV(lasso, param_grid={'alpha': alphas}, cv=5)
grid_search.fit(X_scaled, y)

# Get the best model and print the selected alpha
best_lasso = grid_search.best_estimator_
print("Best Alpha:", best_lasso.alpha)

# Print coefficients
print("Feature Coefficients:")
for i, feature in enumerate(predictors):
    print(f"{feature}: {best_lasso.coef_[i]}")

# Get selected features
selected_lasso_features = [predictors[i] for i in range(len(predictors)) if best_lasso.coef_[i] != 0]
print("Selected LASSO Features:", selected_lasso_features)

