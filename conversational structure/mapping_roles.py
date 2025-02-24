import pandas as pd

# File paths (adjust if necessary)
facilitators_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\facilitators_features_clustered.csv"
participants_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\participants_features_clustered.csv"
conversational_structure_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure.csv"

# Read the CSV files
df_fac = pd.read_csv(facilitators_path)
df_part = pd.read_csv(participants_path)
df_conv = pd.read_csv(conversational_structure_path)

# --- Process Facilitators ---

# Count occurrences of each cluster per conversation_id
fac_counts = df_fac.groupby(['conversation_id', 'cluster']).size().unstack(fill_value=0).reset_index()

# Rename columns: cluster 0 -> Managers, cluster 1 -> Interlocutors
fac_counts = fac_counts.rename(columns={1: 'Managers', 0: 'Interlocutors'})

# --- Process Participants ---

# Count occurrences of each cluster per conversation_id
part_counts = df_part.groupby(['conversation_id', 'cluster']).size().unstack(fill_value=0).reset_index()

# Rename columns: cluster 0 -> Socializers, cluster 1 -> Story tellers, cluster 2 -> Debators
part_counts = part_counts.rename(columns={0: 'Socializers', 1: 'Story tellers', 2: 'Debators'})

# --- Merge Counts into Conversational Structure ---

# Merge facilitators counts
df_merged = pd.merge(df_conv, fac_counts, on='conversation_id', how='left')

# Merge participants counts
df_merged = pd.merge(df_merged, part_counts, on='conversation_id', how='left')

# Replace NaN values with 0 (for conversations where a particular cluster might be missing)
columns_to_fill = ['Managers', 'Interlocutors', 'Socializers', 'Story tellers', 'Debators']
df_merged[columns_to_fill] = df_merged[columns_to_fill].fillna(0)

# Optionally, convert the counts to integers (if needed)
df_merged[columns_to_fill] = df_merged[columns_to_fill].astype(int)

# Save the resulting DataFrame to a new CSV file
output_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure_with_clusters.csv"
df_merged.to_csv(output_path, index=False)
print(df_merged.columns)
print("Merged file saved successfully.")
role_columns = ['Managers', 'Interlocutors', 'Socializers', 'Story tellers', 'Debators']

# Generate descriptive statistics for these columns
role_stats = df_merged[role_columns].describe()
print(role_stats)