import os
import pandas as pd

# Define the folder where the CSV files are located
folder_path = "seed42"

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Initialize an empty list to store dataframes
dfs = []

# Read each CSV file and append it to the list
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all dataframes into one
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv(os.path.join(folder_path, 'hyper_csv_' + folder_path + '.csv'), index=False)

print("CSV files have been combined into 'hyper_csv.csv'")
