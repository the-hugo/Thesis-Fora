import pandas as pd

# Define file paths
file1_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\LIWC-22 Results - data_llama70B_processed_output___ - LIWC Analysis_small.csv"
file2_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\LLM as Judge\merged_data.csv"

# Read CSV files
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Define the unique identifier columns for merging
keys = ['id']

# Identify extra columns in File 1 that are not in File 2.
# We don't add overlapping columns (other than the merge keys).
extra_cols = [col for col in df1.columns if col not in df2.columns]

# Create a reduced DataFrame from File 1 with only the keys and extra columns
df1_reduced = df1[keys + extra_cols]

# Merge File 1 (reduced) into File 2 using an outer join so that all rows are kept.
merged_df = pd.merge(df2, df1_reduced, on=keys, how='outer')

# Optional: Save the merged DataFrame to a new CSV file
merged_df.to_csv("final.csv", index=False)

#print("Merge complete. Merged file saved at:", output_path)
