import pandas as pd

# Load the healthcare dataset
healthcare_data = pd.read_csv('forecasting_data_cleaned.csv')
healthcare_data['Date'] = pd.to_datetime(healthcare_data['Date'])
healthcare_data.set_index('Date', inplace=True)

# Load the weather dataset
weather_data = pd.read_csv('daily_weather_data.csv')
weather_data['DateTime'] = pd.to_datetime(weather_data['DateTime'])
weather_data.set_index('DateTime', inplace=True)

# Join the datasets on the index, which represents the datetime
merged_data = healthcare_data.join(weather_data, how='inner')

# Rename the index to "Date"
merged_data.index.name = "Date"

# Save the merged dataset to a new CSV file
merged_data.to_csv('merged_healthcare_weather_data.csv')

# Print a preview of the resulting dataset
print(merged_data.head())
