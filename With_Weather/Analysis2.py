import pandas as pd

# Function to process individual numeric values
def clean_numeric(item):
    # Check if the value is a string that can be converted to a float
    if isinstance(item, str) and item.replace('.', '', 1).isdigit():
        return round(float(item), 2)
    return item

# Load the dataset with proper parsing
merged_data = pd.read_csv('merged_healthcare_weather_data.csv', converters={
    'TEMP': lambda x: round(float(x), 2) if x.replace('.', '', 1).isdigit() else x,
    'PRCP': lambda x: round(float(x), 2) if x.replace('.', '', 1).isdigit() else x,
    'HMDT': lambda x: round(float(x), 2) if x.replace('.', '', 1).isdigit() else x,
    'WND_SPD': lambda x: round(float(x), 2) if x.replace('.', '', 1).isdigit() else x,
    'ATM_PRESS': lambda x: round(float(x), 2) if x.replace('.', '', 1).isdigit() else x,
}, na_values=['NaN', 'nan', ''])  # Define NaN values handling

# Convert 'Date' or 'DateTime' to datetime
merged_data['Date'] = pd.to_datetime(merged_data['Date'])
merged_data.set_index('Date', inplace=True)

# Save to a new CSV
merged_data.to_csv('enhanced_healthcare_weather_data.csv', index=True)

# Preview
print(merged_data.head())
