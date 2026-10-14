import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def mean_absolute_scaled_error(y_true, y_pred, y_train):
    y_true, y_pred, y_train = np.array(y_true), np.array(y_pred), np.array(y_train)
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum()/(n-1)
    errors = np.abs(y_true - y_pred)
    return errors.mean()/d

