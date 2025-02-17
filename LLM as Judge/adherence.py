import pandas as pd

# 1. Load the adherence scores CSV
adherence_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\adherence_results_classified.csv"
adherence_df = pd.read_csv(adherence_path)

# 2. Calculate the average llm_adherence_score per conversation_id and speaker_name
avg_adherence = (
    adherence_df
    .groupby(['conversation_id', 'speaker_name'])['llm_adherence_score']
    .mean()
    .reset_index()
)
avg_adherence.rename(columns={'llm_adherence_score': 'avg_llm_adherence_score'}, inplace=True)

# 3. Load the "to merge" file (pickle file)
to_merge_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\LLM as Judge\transformed_data.pkl"
to_merge_df = pd.read_pickle(to_merge_path)

# 4. Merge the average adherence scores with the to_merge file on conversation_id and speaker_name
merged_df = pd.merge(to_merge_df, avg_adherence, on=['conversation_id', 'speaker_name'], how='left')

# Optionally, save the merged DataFrame back to disk
merged_df.to_pickle("merged_data.pkl")
merged_df.to_csv("merged_data.csv")

print("Merge complete. The merged DataFrame has been saved to 'merged_data.pkl'.")
