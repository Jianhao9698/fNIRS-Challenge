import pandas as pd


dataPathMainTask = '../../DataChallenge2024_forStudents/data/MainTask/'
csvWithoutHeader = 'SPNDataChallenge0001.csv'

# Define the column names for the CSV file
column_names = [
    "SUBJECT",                # 1
    "SESSION",                # 2
    "DATASOURCE",            # 3
    "STRUCTUREDDATA",        # 4
    "CHANNEL",               # 5
    "SIGNAL",                # 6
    "STIMULUS",              # 7
    "BLOCK",                 # 8
    "MEAN_BASELINE",         # 9
    "STD_BASELINE",          # 10
    "MEAN_TASK",             # 11
    "STD_TASK",              # 12
    "TASK_MINUS_BASELINE",   # 13
    "AREA_UNDER_CURVE_TASK", # 14
    "TIME_TO_PEAK",          # 15
    "TIME_TO_NADIR"          # 16
]

df = pd.read_csv(
    dataPathMainTask+csvWithoutHeader,
    header=None,         # File without header
    names=column_names,  # Add the header
    sep=",",             # Default separator is comma
)

# Check the info of the DataFrame
print("DataFrame Info:")
print(df.info(), '\n', df.shape)
# First 5 rows of the DataFrame
print("Preview of the first 5 rows:")
print(df.head())

df.dropna(how='all', inplace=True)
df.drop_duplicates(inplace=True)          # Remove duplicate rows

# Resort the DataFrame by "SUBJECT", "SESSION", "CHANNEL"
df.sort_values(by=["SUBJECT", "SESSION", "CHANNEL"], inplace=True)
df.reset_index(drop=True, inplace=True)

# Save the DataFrame to a new CSV file
print(f"Processed DataFrame info: {df.info()}, \n{df.shape}")
df.to_csv("SPNDataChallenge0001_withHeader.csv", index=False)

print("\nSuccessfully added headers and saved to 'SPNDataChallenge0001_withHeader.csv'.")
