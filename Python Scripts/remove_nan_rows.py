import pandas as pd

# Load the original dataset
file_path = "/content/sample_data/exoplanet_database_complete.csv"  # Adjust the path if necessary
df = pd.read_csv(file_path, delimiter=";")

# Strip extra spaces in column names
df.columns = df.columns.str.strip()

# Function to remove the row with the most NaNs for each planet
def remove_row_with_most_nans(group):
    # Calculate the number of NaNs per row (across all columns)
    nan_counts = group.isna().sum(axis=1)

    # Find the index of the row with the most NaNs
    row_with_most_nans = nan_counts.idxmax()

    # Drop the row with the most NaNs
    group_cleaned = group.drop(index=row_with_most_nans)

    return group_cleaned

# Group by 'planet_name' and apply the function to each group
df_cleaned = df.groupby('planet_name', group_keys=False).apply(remove_row_with_most_nans)

# Remove any remaining duplicates by keeping the first occurrence of each planet
df_cleaned = df_cleaned.drop_duplicates(subset=["planet_name"], keep="first")

# Reset the index after dropping rows
df_cleaned.reset_index(drop=True, inplace=True)

# Display the cleaned dataframe
print(f"Number of rows before cleaning: {df.shape[0]}")
print(f"Number of rows after cleaning: {df_cleaned.shape[0]}")
print(df_cleaned.head())
