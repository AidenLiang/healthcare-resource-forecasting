import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import numpy as np
# from joblib import Parallel, delayed  # Uncomment for parallel processing

# Load the dataset
data = pd.read_csv('forecasting_data_cleaned.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Convert to time series
data.index = data.index.to_period('D')
time_series = data['ICU Beds in Use - Total']

# Split the data into training and testing sets
train_size = int(len(time_series) * 0.8)
train, test = time_series[:train_size], time_series[train_size:]

# Define parameter ranges
p = q = P = Q = range(0, 3)
d = D = [0, 1]
s = [7, 14]

# Generate all combinations
param_combinations = list(itertools.product(p, d, q))
seasonal_combinations = list(itertools.product(P, D, Q, s))

# Track the best model configuration
best_aic = np.inf
best_params = None
best_seasonal_params = None

# Define a convergence threshold for early stopping
no_improvement_limit = 5
no_improvement_count = 0

# Define the fitting function
def fit_model(params, seasonal_params):
    try:
        model = SARIMAX(train, order=params, seasonal_order=seasonal_params)
        result = model.fit()
        return (result.aic, params, seasonal_params)
    except Exception as e:
        return (np.inf, None, None)

# Option 1: Sequential processing
for params in param_combinations:
    for seasonal_params in seasonal_combinations:
        aic, p, s = fit_model(params, seasonal_params)

        if aic < best_aic:
            best_aic = aic
            best_params = p
            best_seasonal_params = s
            no_improvement_count = 0
        else:
            no_improvement_count += 1

            if no_improvement_count >= no_improvement_limit:
                break
    if no_improvement_count >= no_improvement_limit:
        break

# Option 2: Parallel processing
# results = Parallel(n_jobs=-1)(delayed(fit_model)(p, s) for p in param_combinations for s in seasonal_combinations)
# best_result = min(results, key=lambda x: x[0])
# best_aic, best_params, best_seasonal_params = best_result

# Output the best configuration
print(f'Best Params: {best_params}')
print(f'Best Seasonal Params: {best_seasonal_params}')
print(f'Best AIC: {best_aic}')