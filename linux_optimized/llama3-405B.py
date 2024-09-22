import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from huggingface_hub import login
import huggingface_hub.utils
import pickle

login("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")
huggingface_hub.utils._http.default_timeout = 230
logging.set_verbosity_error()

collection_name = "collection-150_Maine"
input_path = "/mounts/Users/cisintern/pfromm/{}_transformed_data.csv".format(collection_name)
data = pd.read_csv(input_path, sep=',')

extractor = pipeline("feature-extraction", model="meta-llama/Meta-Llama-3.1-8B")
ner_pipeline = pipeline("ner", model="meta-llama/Meta-Llama-3.1-8B", aggregation_strategy="simple")

data['CLS_Embedding'] = None
data['Sentence_NER'] = None
data['Sentence_Sentiment'] = None

# Define functions
def feature_extraction(text):
    features = extractor(text)[0]
    features_tensor = torch.tensor(features)
    sentence_embedding = features_tensor[0].tolist()
    return sentence_embedding

def ner(text):
    return ner_pipeline(text)

# Function to process a single group (i.e., sentence)
def process_sentence(snippet):
    sentence = snippet['Content']
    if not isinstance(sentence, str) or sentence.strip() == "":
        return None  # Return None for invalid or empty sentences

    features = feature_extraction(sentence)
    ner_results = ner(sentence)
    
    return {
        'features': features,
        'ner_results': ner_results,
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
            with open(final_output_path, 'wb') as f:
                pickle.dump(data, f)
            current_conv = row['Conversation ID']
            print("Processing", current_conv)

        result = process_sentence(row)
        if result:
            data.at[index, 'CLS_Embedding'] = result['features']
            data.at[index, 'Sentence_NER'] = result['ner_results']
        else:
            data.at[index, 'CLS_Embedding'] = None
            data.at[index, 'Sentence_NER'] = None

final_output_path = "/mounts/Users/cisintern/pfromm/{collection}_llama3_processed_output.pkl".format(collection=collection_name)
parallel_processing(data)

with open(final_output_path, 'wb') as f:
    pickle.dump(data, f)
print(f"Final output saved to {final_output_path}")
