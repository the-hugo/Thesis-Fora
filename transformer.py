import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Disable oneDNN optimizations (if needed)
logging.set_verbosity_error()

# Load the CSV data
data = pd.read_csv("./data/transformed_output.csv", sep=',')

# Initialize transformers pipelines for different tasks
extractor = pipeline("feature-extraction", model="bert-base-uncased")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define functions
def feature_extraction(text):
    return extractor(text)[0]

def ner(text):
    return ner_pipeline(text)

def sentiment_analysis(text):
    return sentiment_classifier(text)[0]

# Function to process a single group (i.e., sentence)
def process_sentence(group):
    sentence = group['Content'].iloc[0] 
    if not isinstance(sentence, str) or sentence.strip() == "":
        return None  # Return None for invalid or empty sentences

    features = feature_extraction(sentence)
    ner_results = ner(sentence)
    sentiment = sentiment_analysis(sentence)
    
    return {
        'features': features,
        'ner_results': ner_results,
        'sentiment': sentiment
    }

data = data.drop_duplicates(subset=['Index in Conversation', 'Content'])

# Group by Conversation ID and Speaker ID, as we need to distinguish between conversations and speakers
grouped_data = data.groupby(['Conversation ID', 'Speaker ID', 'Index in Conversation'])

# Add new columns to the original data to hold NLP results
data['Sentence_Features'] = None
data['Sentence_NER'] = None
data['Word_Sentiment'] = None

# Function to handle parallel processing
def parallel_processing(grouped_data, batch_size=1000, checkpoint_interval=1000):
    temp_results = []  # Temporary storage for results
    futures = []

    # Load checkpointed data if it exists
    checkpoint_path = "./data/checkpoint_output.csv"
    if os.path.exists(checkpoint_path):
        processed_data = pd.read_csv(checkpoint_path)
        start_idx = len(processed_data)
        print(f"Resuming from checkpoint at index {start_idx}")
    else:
        processed_data = pd.DataFrame()  # Initialize as an empty DataFrame if no checkpoint exists
        start_idx = 0

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on your system
        grouped_data_tqdm = tqdm(enumerate(grouped_data), total=len(grouped_data), desc="Processing Groups")
        for idx, (name, group) in grouped_data_tqdm:
            if idx < start_idx:  # Skip already processed data
                continue
            # Submit task to executor
            futures.append(executor.submit(process_sentence, group))
            
            # Collect results every `checkpoint_interval`
            if (idx + 1) % checkpoint_interval == 0:
                for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting Results"):
                    result = future.result()
                    if result:
                        temp_results.append({
                            'Conversation ID': name[0],
                            'Speaker ID': name[1],
                            'Index in Conversation': name[2],
                            'features': result['features'],
                            'ner_results': result['ner_results'],
                            'sentiment': result['sentiment']['label']
                        })
                    else:
                        print("HI")
                        # Add placeholders for skipped sentences
                        temp_results.append({
                            'Conversation ID': name[0],
                            'Speaker ID': name[1],
                            'Index in Conversation': name[2],
                            'features': None,
                            'ner_results': None,
                            'sentiment': None
                        })

                # After processing, directly update the corresponding rows in `data` DataFrame
                for temp_result in temp_results:
                    mask = (
                        (data['Conversation ID'] == temp_result['Conversation ID']) &
                        (data['Speaker ID'] == temp_result['Speaker ID']) &
                        (data['Index in Conversation'] == temp_result['Index in Conversation'])
                    )
                    
                    # Use `.at[]` for scalar values to avoid dimension mismatch issues
                    idx_to_update = data[mask].index
                    
                    if len(idx_to_update) == 1:  # Ensure we're updating one row
                        data.at[idx_to_update[0], 'Sentence_Features'] = temp_result['features']
                        data.at[idx_to_update[0], 'Sentence_NER'] = temp_result['ner_results']
                        data.at[idx_to_update[0], 'Word_Sentiment'] = temp_result['sentiment']
                    else:
                        print(f"Warning: Multiple rows found for the same index: {temp_result['Index in Conversation']}")

                # Save the updated DataFrame with checkpoint
                processed_data = pd.concat([processed_data, pd.DataFrame(temp_results)], ignore_index=True)
                processed_data.to_csv(checkpoint_path, index=False)
                print(f"Checkpoint saved at sentence {idx + 1}")
                
                # Reset for the next batch
                temp_results = []
                futures.clear()  # Clear the list of futures
                
        # Handle any remaining futures
        for future in tqdm(as_completed(futures), total=len(futures), desc="Final Results"):
            result = future.result()
            if result:
                temp_results.append({
                    'Conversation ID': name[0],
                    'Speaker ID': name[1],
                    'Index in Conversation': name[2],
                    'features': result['features'],
                    'ner_results': result['ner_results'],
                    'sentiment': result['sentiment']['label']
                })
            else:
                # Add placeholders for skipped sentences
                temp_results.append({
                    'Conversation ID': name[0],
                    'Speaker ID': name[1],
                    'Index in Conversation': name[2],
                    'features': None,
                    'ner_results': None,
                    'sentiment': None
                })

        # Add final batch results directly to the original DataFrame
        for temp_result in temp_results:
            mask = (
                (data['Conversation ID'] == temp_result['Conversation ID']) &
                (data['Speaker ID'] == temp_result['Speaker ID']) &
                (data['Index in Conversation'] == temp_result['Index in Conversation'])
            )
            
            idx_to_update = data[mask].index
            if len(idx_to_update) == 1:  # Ensure we're updating one row
                data.at[idx_to_update[0], 'Sentence_Features'] = temp_result['features']
                data.at[idx_to_update[0], 'Sentence_NER'] = temp_result['ner_results']
                data.at[idx_to_update[0], 'Word_Sentiment'] = temp_result['sentiment']

        # Save the final batch to the checkpoint file
        processed_data = pd.concat([processed_data, pd.DataFrame(temp_results)], ignore_index=True)
        processed_data.to_csv(checkpoint_path, index=False)
        print(f"Final checkpoint saved.")

# Run the parallel processing function
parallel_processing(grouped_data)

# Save the final output
final_output_path = "./data/processed_output.csv"
data.to_csv(final_output_path, index=False)
print(f"Final output saved to {final_output_path}")
                             