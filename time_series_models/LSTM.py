import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from metrics import mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error, mean_absolute_scaled_error


# Load the dataset
data = pd.read_csv('forecasting_data_cleaned.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
values = data['ICU Beds in Use - Total'].values.reshape(-1, 1)

# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)

# Convert to supervised learning problem
def create_dataset(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        X.append(a)
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 1
X, y = create_dataset(scaled_values, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into train and test sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
train_X, test_X = X[0:train_size], X[train_size:len(X)]
train_y, test_y = y[0:train_size], y[train_size:len(y)]

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(train_X, train_y, epochs=100, batch_size=1, verbose=2)

# Predict
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

# Invert predictions back to original scale
train_predict = scaler.inverse_transform(train_predict)
train_y = scaler.inverse_transform([train_y])
test_predict = scaler.inverse_transform(test_predict)
test_y = scaler.inverse_transform([test_y])

# Calculate RMSE
train_score = np.sqrt(mean_squared_error(train_y[0], train_predict[:,0]))
print('Train Score: %.2f RMSE' % (train_score))
test_score = np.sqrt(mean_squared_error(test_y[0], test_predict[:,0]))
print('Test Score: %.2f RMSE' % (test_score))
# Calculate additional metrics
mae_train = mean_absolute_error(train_y[0], train_predict[:,0])
mape_train = mean_absolute_percentage_error(train_y[0], train_predict[:,0])
smape_train = symmetric_mean_absolute_percentage_error(train_y[0], train_predict[:,0])
mase_train = mean_absolute_scaled_error(train_y[0], train_predict[:,0], train_y[0])

mae_test = mean_absolute_error(test_y[0], test_predict[:,0])
mape_test = mean_absolute_percentage_error(test_y[0], test_predict[:,0])
smape_test = symmetric_mean_absolute_percentage_error(test_y[0], test_predict[:,0])
mase_test = mean_absolute_scaled_error(test_y[0], test_predict[:,0], train_y[0])

# Print out metrics
print('Train MAE: %.2f' % (mae_train))
print('Train MAPE: %.2f' % (mape_train))
print('Train sMAPE: %.2f' % (smape_train))
print('Train MASE: %.2f' % (mase_train))

print('Test MAE: %.2f' % (mae_test))
print('Test MAPE: %.2f' % (mape_test))
print('Test sMAPE: %.2f' % (smape_test))
print('Test MASE: %.2f' % (mase_test))