import pandas as pd

# Load the dataset
weather_data = pd.read_csv('data.csv')

# Combine YEAR, MO, DY, and HR into a single datetime column
weather_data['DateTime'] = pd.to_datetime(weather_data[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')

# Drop the individual columns to clean up the dataset
weather_data.drop(columns=['YEAR', 'MO', 'DY', 'HR'], inplace=True)

# Set DateTime as the index
weather_data.set_index('DateTime', inplace=True)

# Resample data to daily values using mean aggregation
daily_weather_data = weather_data.resample('D').mean()

# Round off the numerical data to two decimal places
daily_weather_data = daily_weather_data.round(2)

# Save the daily dataset to a new CSV file
daily_weather_data.to_csv('daily_weather_data.csv')

# Print a preview of the resulting dataset
print(daily_weather_data.head())
