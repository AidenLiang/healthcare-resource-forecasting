import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset before and after cleaning
data_before = pd.read_csv('forecasting_data.csv')
data_after = pd.read_csv('forecasting_data_cleaned.csv')

# Convert the 'Date' column to datetime for both datasets
data_before['Date'] = pd.to_datetime(data_before['Date'])
data_after['Date'] = pd.to_datetime(data_after['Date'])

# Set the 'Date' as the index for easier plotting
data_before.set_index('Date', inplace=True)
data_after.set_index('Date', inplace=True)

# Identify the removed rows
removed_data = data_before[~data_before.index.isin(data_after.index)]

# Define a function to plot a specific column's time series
def plot_column_time_series(column_name):
    plt.figure(figsize=(12, 6))

    # Plot the data before cleaning
    plt.plot(data_before.index, data_before[column_name], label='Before Cleaning', color='blue')

    # Plot the data after cleaning
    plt.plot(data_after.index, data_after[column_name], label='After Cleaning', color='green')

    # Plot the removed rows
    plt.scatter(removed_data.index, removed_data[column_name], label='Removed Data', color='red', s=10)

    plt.title(f"Time Series for {column_name}")
    plt.xlabel("Date")
    plt.ylabel(column_name)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

# Choose columns to visualize
columns_to_visualize = [
    'Ventilators - Total Capacity',
    'Ventilators in Use - Total',
    'Ventilators in Use - COVID-19',
    'ICU Beds - Total Capacity',
    'ICU Beds in Use - Total',
    'ICU Beds in Use - COVID-19',
    'Acute Non-ICU Beds - Total Capacity',
    'Acute Non-ICU Beds in Use - Total',
    'Acute Non-ICU Beds in Use - COVID-19',
    'Combined Hospital Beds in Use - COVID-19'
]

# Plot each selected column
for column in columns_to_visualize:
    plot_column_time_series(column)
