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
data = pd.read_csv('forecasting_data_cleaned.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Create new features from the Date
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data.set_index('Date', inplace=True)

# Select features and target
target = 'ICU Beds in Use - Total'
features = data.columns.drop(target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

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

    # Line plot for the actual and predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.reset_index(drop=True), label='Actual')
    plt.plot(predictions, label='Predicted', alpha=0.7)
    plt.xlabel('Test Case Index')
    plt.ylabel('ICU Beds in Use')
    plt.title('Actual vs. Predicted ICU Beds in Use Over Test Set (Decision Tree)')
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

