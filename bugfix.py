import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Disable oneDNN optimizations (if needed)
logging.set_verbosity_error()

# Load the CSV data
data = pd.read_csv("./data/scraped_sampled.csv", sep=';')

# Initialize transformers pipelines for different tasks
extractor = pipeline("feature-extraction", model="bert-base-uncased")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

data = data.drop_duplicates(subset=['Audio Start Offset', 'Content'])
data = data.drop(columns=['Start', 'End', 'Duration'])

# Add new columns to the original data to hold NLP results
data['Sentence_Features'] = None
data['Sentence_NER'] = None
data['Word_Sentiment'] = None

# Define functions
def feature_extraction(text):
    return extractor(text)[0]

def ner(text):
    return ner_pipeline(text)

def sentiment_analysis(text):
    return sentiment_classifier(text)[0]

# Function to process a single group (i.e., sentence)
def process_sentence(group):
    sentence = group['Content']
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

# Function to handle parallel processing
def parallel_processing(grouped_data):
    current_conv = None
    global final_output_path

    for index, row in tqdm(grouped_data.iterrows(), total=len(grouped_data)):
        if current_conv is None:
            current_conv = row['Conversation ID']
        if current_conv != row['Conversation ID']:
            print("Saved", current_conv)
            data.to_csv(final_output_path, mode='a', header=not os.path.exists(final_output_path), index=False)
            current_conv = row['Conversation ID']
            print("Processing", current_conv)

        result = process_sentence(row)
        if result:
            data.at[index, 'Sentence_Features'] = result['features']
            data.at[index, 'Sentence_NER'] = result['ner_results']
            data.at[index, 'Word_Sentiment'] = result['sentiment']['label']
        else:
            data.at[index, 'Sentence_Features'] = None
            data.at[index, 'Sentence_NER'] = None
            data.at[index, 'Word_Sentiment'] = None

final_output_path = "./data/processed_output.csv"
parallel_processing(data)

data.to_csv(final_output_path, index=False)
print(f"Final output saved to {final_output_path}")
