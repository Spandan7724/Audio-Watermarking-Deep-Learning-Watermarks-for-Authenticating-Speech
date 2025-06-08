import pandas as pd

# Load the CSV files
df1 = pd.read_csv("dataset_creation/200_hours_metadata.csv")
df2 = pd.read_csv("subset_100hours.csv")

# Extract the 'filepath' column as sets
filepaths1 = set(df1['filepath'])
filepaths2 = set(df2['filepath'])

# Compute the intersection of the two sets
common_filepaths = filepaths1.intersection(filepaths2)

# Print the number of common file paths
print("Number of files present in both CSVs:", len(common_filepaths))

# Optional: print the common file paths if needed
print("Common file paths:", common_filepaths)
