import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Disable oneDNN optimizations (if needed)
logging.set_verbosity_error()

# Load the CSV data
data = pd.read_csv("./data/transformed_output.csv")

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

# Group by Conversation ID and Speaker ID, as we need to distinguish between conversations and speakers
grouped_data = data.groupby(['Conversation ID', 'Speaker ID', 'Index in Conversation'])

# Load checkpointed data if it exists
checkpoint_path = "./data/checkpoint_output.csv"
if os.path.exists(checkpoint_path):
    processed_data = pd.read_csv(checkpoint_path)
    start_idx = len(processed_data)
    print(f"Resuming from checkpoint at index {start_idx}")
else:
    processed_data = pd.DataFrame(columns=data.columns)
    start_idx = 0

# Create lists to store results
sentence_features = []
sentence_ner = []
sentence_sentiment = []

def parallel_processing(grouped_data, batch_size=1000, checkpoint_interval=1000):
    global sentence_features, sentence_ner, sentence_sentiment
    
    temp_results = []  # Temporary storage for results
    futures = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
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
                            'Sentence_Features': result['features'],
                            'Sentence_NER': result['ner_results'],
                            'Word_Sentiment': result['sentiment']['label']
                        })
                    else:
                        # Add placeholders for skipped sentences
                        temp_results.append({
                            'Conversation ID': name[0],
                            'Speaker ID': name[1],
                            'Index in Conversation': name[2],
                            'Sentence_Features': None,
                            'Sentence_NER': None,
                            'Word_Sentiment': None
                        })
                
                # Save checkpoint
                temp_df = pd.DataFrame(temp_results)
                data.update(temp_df.set_index(['Conversation ID', 'Speaker ID', 'Index in Conversation']))
                
                # Append to checkpoint data
                processed_data = pd.concat([processed_data, temp_df])
                processed_data.to_csv(checkpoint_path, index=False)
                print(f"Checkpoint saved at sentence {idx + 1}")
                
                # Reset for the next batch
                temp_results = []
                futures = []  # Reset futures for next batch
                
        # Handle any remaining futures
        for future in tqdm(as_completed(futures), total=len(futures), desc="Final Results"):
            result = future.result()
            if result:
                temp_results.append({
                    'Conversation ID': name[0],
                    'Speaker ID': name[1],
                    'Index in Conversation': name[2],
                    'Sentence_Features': result['features'],
                    'Sentence_NER': result['ner_results'],
                    'Word_Sentiment': result['sentiment']['label']
                })
            else:
                # Add placeholders for skipped sentences
                temp_results.append({
                    'Conversation ID': name[0],
                    'Speaker ID': name[1],
                    'Index in Conversation': name[2],
                    'Sentence_Features': None,
                    'Sentence_NER': None,
                    'Word_Sentiment': None
                })
                
        # Add final batch to data
        if temp_results:
            final_df = pd.DataFrame(temp_results)
            data.update(final_df.set_index(['Conversation ID', 'Speaker ID', 'Index in Conversation']))
            processed_data = pd.concat([processed_data, final_df])
            processed_data.to_csv(checkpoint_path, index=False)
            print(f"Final checkpoint saved.")

# Run the parallel processing with checkpointing
parallel_processing(grouped_data)

# After processing, save the final data
data['Sentence_Features'] = sentence_features
data['Sentence_NER'] = sentence_ner
data['Word_Sentiment'] = [sent['label'] if sent else None for sent in sentence_sentiment]

# Save the final processed dataframe
final_output_path = "./data/processed_output.csv"
data.to_csv(final_output_path, index=False)
print(f"Final output saved to {final_output_path}")
