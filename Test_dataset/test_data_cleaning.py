import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the CSV file is in the same directory as your script
file_name = 'COVID-19_Hospital_Capacity_Metrics_-_Historical.csv'



# Define the columns that are relevant for healthcare forecasting
forecasting_columns = [
    'Date',
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

# Read the CSV file using the defined columns
data = pd.read_csv(file_name, usecols=forecasting_columns)
print(data.columns)
data['Date'] = pd.to_datetime(data['Date'])  # Convert the date column to datetime

# Set the date as the index to facilitate time-based interpolation
data.set_index('Date', inplace=True)

# Fill missing values using time-based interpolation
data = data.interpolate(method='time')

# Sort the DataFrame by the 'Date' index (to ensure proper order for time series analysis)
data.sort_index(inplace=True)

# Save the selected columns to a new CSV file for further analysis
data.to_csv('forecasting_data.csv')

print("Data with selected columns saved to 'forecasting_data.csv'.")

# Remove outliers using IQR
def remove_outliers(df, column_names):
    for column in column_names:
        if column == 'Date':  # Skip processing the Date column
            continue
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

data = remove_outliers(data, forecasting_columns)

# Check for negative values and set them to NaN
for column in forecasting_columns[1:]:
    data[column] = data[column].apply(lambda x: x if x >= 0 else None)

# Ensure no ICU beds in use exceed total ICU capacity
data['ICU Beds in Use - Total'] = data.apply(
    lambda x: min(x['ICU Beds in Use - Total'], x['ICU Beds - Total Capacity']), axis=1
)

# Save the cleaned data
data.to_csv('forecasting_data_cleaned.csv')
print("Cleaned data saved to 'forecasting_data_cleaned.csv'.")

# Load data before and after cleaning
data_before_cleaning = pd.read_csv('forecasting_data.csv', usecols=forecasting_columns)
data_before_cleaning['Date'] = pd.to_datetime(data_before_cleaning['Date'])
data_before_cleaning.set_index('Date', inplace=True)

data_after_cleaning = pd.read_csv('forecasting_data_cleaned.csv', usecols=forecasting_columns)
data_after_cleaning['Date'] = pd.to_datetime(data_after_cleaning['Date'])
data_after_cleaning.set_index('Date', inplace=True)

# Define functions for plotting comparison and histograms
def plot_comparison(data_before, data_after, column_names):
    for column in column_names[1:]:
        plt.figure(figsize=(12, 6))

        combined_data = pd.DataFrame({
            'Before': data_before[column],
            'After': data_after[column]
        })

        sns.boxplot(data=combined_data)
        plt.title(f'Box Plot Comparison for {column}')
        plt.xticks(rotation=45)
        plt.show()

def plot_histograms(data_before, data_after, column_names):
    for column in column_names[1:]:
        plt.figure(figsize=(12, 6))

        sns.histplot(data_before[column], color="blue", label="Before Cleaning", kde=True, element="step", stat="density")
        sns.histplot(data_after[column], color="red", label="After Cleaning", kde=True, element="step", stat="density")

        plt.title(f'Histogram for {column}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

# Call the plotting functions
plot_comparison(data_before_cleaning, data_after_cleaning, forecasting_columns)
plot_histograms(data_before_cleaning, data_after_cleaning, forecasting_columns)
