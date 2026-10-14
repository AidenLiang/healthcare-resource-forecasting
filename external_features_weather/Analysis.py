import pandas as pd

# Load the dataset with correct parsing settings
merged_data = pd.read_csv('merged_healthcare_weather_data.csv', decimal=".", na_values=['NaN', 'nan', ''])  # Define decimal points and NaN handling

# Convert 'Date' or 'DateTime' to datetime
merged_data['Date'] = pd.to_datetime(merged_data['Date'])
merged_data.set_index('Date', inplace=True)

# Round numeric columns to manage precision and avoid floating-point issues
numeric_cols = ['TEMP', 'PRCP', 'HMDT', 'WND_SPD', 'ATM_PRESS']
merged_data[numeric_cols] = merged_data[numeric_cols].apply(lambda x: x.round(2))

# Additional feature engineering, differences, rolling metrics
merged_data['TEMP_DIFF'] = merged_data['TEMP'].diff()
merged_data['PRCP_DIFF'] = merged_data['PRCP'].diff()
merged_data['HMDT_DIFF'] = merged_data['HMDT'].diff()
merged_data['WND_SPD_DIFF'] = merged_data['WND_SPD'].diff()

# Rolling metrics
merged_data['TEMP_ROLL_MEAN_7'] = merged_data['TEMP'].rolling(7).mean()
merged_data['TEMP_ROLL_VAR_7'] = merged_data['TEMP'].rolling(7).var()

# Combined metrics
merged_data['TEMP_HUMIDITY_COMBO'] = merged_data['TEMP'] * merged_data['HMDT']

# Drop rows with any NaN values
merged_data.dropna(inplace=True)

# Save to a new CSV
merged_data.to_csv('enhanced_healthcare_weather_data.csv', index=True)

# Preview
print(merged_data[['TEMP_DIFF', 'TEMP_ROLL_MEAN_7', 'TEMP_HUMIDITY_COMBO']].head())
