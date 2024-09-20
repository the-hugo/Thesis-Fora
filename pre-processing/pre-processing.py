import pandas as pd
import json

# Load the CSV file
input_csv = 'C:\\Users\\paul-\\Documents\\Uni\\Management and Digital Technologies\\Thesis Fora\\data\\input\\collection-150_Maine\\collection-150_Maine_data.csv'
output_csv = 'C:\\Users\\paul-\\Documents\\Uni\\Management and Digital Technologies\\Thesis Fora\\data\\input\\collection-150_Maine\\collection-150_Maine_transformed_data.csv'

# Read the data from the CSV into a DataFrame
df = pd.read_csv(input_csv)

# Initialize a list to collect rows
rows = []

# Loop through each row in the DataFrame and parse the JSON data
for i, row in df.iterrows():
    # Parse each "data" field into a valid JSON object
    entry = json.loads(row["data"])  # Assuming each 'data' field contains valid JSON
    if "snippets" in entry["data"]["entities"]:
        for snippet_id, snippet_data in entry["data"]["entities"]["snippets"].items():
            for word_data in snippet_data["words"]:
                row_data = {
                    "Conversation ID": row,
                    "Snippet ID": snippet_id,        # Key from entities
                    "Conversation ID": snippet_data["conversation_id"],
                    "Audio Start Offset": snippet_data["audio_start_offset"],
                    "Audio End Offset": snippet_data["audio_end_offset"],
                    "Speaker ID": snippet_data["speaker_id"],
                    "Speaker Name": snippet_data["speaker_name"],
                    "Is Facilitator": snippet_data["is_facilitator"],
                    "Index in Conversation": snippet_data["index_in_conversation"],
                    "duration": snippet_data["duration"],
                    "speaker_gender": snippet_data["speaker_gender"],
                    "Content": snippet_data["content"],
                }
                rows.append(row_data)


# Creating a DataFrame from the rows list
df_transformed = pd.DataFrame(rows)
# kill all duplicates
df_transformed.drop_duplicates(inplace=True)

# Save the transformed data to a new CSV file
df_transformed.to_csv(output_csv, index=False)

print(f"Transformation complete. Data saved to {output_csv}")
