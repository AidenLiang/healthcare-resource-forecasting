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
data['Date'] = pd.to_datetime(data['Date'])  # Convert the date column to datetime

# Sort the DataFrame by the 'date' column
data.sort_values('Date', inplace=True)

# Save the selected columns to a new CSV file for further analysis
data.to_csv('forecasting_data.csv', index=False)

print("Data with selected columns saved to 'forecasting_data.csv'.")

# Extract the data for the columns of interest
forecasting_data = data[forecasting_columns]

# Save the data into a list of dictionaries for each row (optional)
list_of_dicts = forecasting_data.to_dict(orient='records')
# test use: print(list_of_dicts)

# Or, if prefer to have separate lists for each column, you can do:
columns_data = {col: forecasting_data[col].tolist() for col in forecasting_columns}

# Fill missing values with the median or mean of the column
data.fillna(data.median(), inplace=True)

# Function to remove outliers using IQR
def remove_outliers(df, column_names):
    for column in column_names:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out the outliers
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Apply the function to your data
data = remove_outliers(data, forecasting_columns)

# Check for negative values and set them to NaN or a threshold
for column in forecasting_columns[1:]:
    data[column] = data[column].apply(lambda x: x if x >= 0 else None)

# Ensure no ICU beds in use exceed total ICU capacity
data['ICU Beds in Use - Total'] = data.apply(
    lambda x: min(x['ICU Beds in Use - Total'], x['ICU Beds - Total Capacity']), axis=1
)

# Save the clean data
data.to_csv('forecasting_data_cleaned.csv', index=False)
print("Cleaned data saved to 'forecasting_data_cleaned.csv'.")

# Plotting functions assume the date column is present to properly index the plots
data.set_index('Date', inplace=True)  # Set the date as the index for easier plotting
data_before_cleaning = pd.read_csv('forecasting_data.csv', usecols=forecasting_columns)
data_before_cleaning.set_index('Date', inplace=True)
data_after_cleaning = pd.read_csv('forecasting_data_cleaned.csv', usecols=forecasting_columns)
data_after_cleaning.set_index('Date', inplace=True)


# Define a function to plot box plots before and after cleaning
def plot_comparison(data_before, data_after, column_names):
    for column in column_names[1:]:
        plt.figure(figsize=(12, 6))
        
        # Creating a DataFrame for plotting
        combined_data = pd.DataFrame({
            'Before': data_before[column],
            'After': data_after[column]
        })
        
        # Plotting
        sns.boxplot(data=combined_data)
        plt.title(f'Box Plot Comparison for {column}')
        plt.show()

# Assume `data_before_cleaning` is your data before removing outliers and `data` is after cleaning
data_before_cleaning = pd.read_csv('forecasting_data.csv', usecols=forecasting_columns)
data_after_cleaning = pd.read_csv('forecasting_data_cleaned.csv', usecols=forecasting_columns)

# Call the function
plot_comparison(data_before_cleaning, data_after_cleaning, forecasting_columns)

def plot_histograms(data_before, data_after, column_names):
    for column in column_names[1:]:
        plt.figure(figsize=(12, 6))
        
        # Plot before cleaning
        sns.histplot(data_before[column], color="blue", label="Before Cleaning", kde=True, element="step", stat="density")
        
        # Plot after cleaning
        sns.histplot(data_after[column], color="red", label="After Cleaning", kde=True, element="step", stat="density")
        
        plt.title(f'Histogram for {column}')
        plt.legend()
        plt.show()

# Call the histogram plotting function
plot_histograms(data_before_cleaning, data_after_cleaning, forecasting_columns)

