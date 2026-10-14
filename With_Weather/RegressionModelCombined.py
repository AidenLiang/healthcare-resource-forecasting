import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the cleaned dataset
data = pd.read_csv('enhanced_healthcare_weather_data.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Create new features from the Date
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data.set_index('Date', inplace=True)

# Select features and target
target = 'ICU Beds in Use - Total'
features = data.columns.drop(target)

# Determine the split point for the last 20% of data
split_point = int(len(data) * 0.8)
X_train, X_test = data[features].iloc[:split_point], data[features].iloc[split_point:]
y_train, y_test = data[target].iloc[:split_point], data[target].iloc[split_point:]

# Helper function for plotting and evaluating models
def evaluate_model(model, model_name, X_train, X_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    predictions = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)

    print(f'{model_name} RMSE: {rmse}')
    print(f'{model_name} MAE: {mae}')
    print(f'{model_name} R-squared: {r_squared}')

    # Line plot: actual vs. predicted values over time
    plt.figure(figsize=(10, 6))
    plt.plot(y_train.index, y_train, label='Training', color='blue')
    plt.plot(y_test.index, y_test, label='Actual Test', color='green')
    plt.plot(y_test.index, predictions, label='Predicted', alpha=0.7, color='red')
    plt.xlabel('Date')
    plt.ylabel('ICU Beds in Use')
    plt.title(f'Actual vs. Predicted ICU Beds in Use Over Time ({model_name})')
    plt.legend()
    plt.show()

    # Scatter plot: actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual ICU Beds in Use')
    plt.ylabel('Predicted ICU Beds in Use')
    plt.title(f'Actual vs. Predicted ICU Beds in Use ({model_name})')
    plt.show()

# List of models and their configurations
models = [
    (BayesianRidge(), "Bayesian Ridge"),
    (Lasso(max_iter=10000), "LASSO"),
    (Ridge(), "Ridge"),
    (DecisionTreeRegressor(random_state=42), "Decision Tree"),
    (LinearRegression(), "Multiple Linear Regression"),
    (LinearRegression(), "Polynomial Regression"),
    (RandomForestRegressor(n_estimators=100, random_state=100), "Random Forest"),
    (SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1), "Support Vector Regression")
]

# Evaluate each model
for model, model_name in models:
    if model_name == "Polynomial Regression":
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        evaluate_model(model, model_name, X_train_poly, X_test_poly)
    elif model_name == "Support Vector Regression":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        evaluate_model(model, model_name, X_train_scaled, X_test_scaled)
    else:
        evaluate_model(model, model_name, X_train, X_test)

