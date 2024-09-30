from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
from transformers import logging, AutoModel
import os
import huggingface_hub.utils
import pickle

logging.set_verbosity_error()

huggingface_hub.utils._http.default_timeout = 230
collection_name = "corpus_data"
input_path = "/mounts/Users/cisintern/pfromm/{}_transformed_data.csv".format(collection_name)
print(f"Loading data from {input_path}")
data = pd.read_csv(input_path, sep=',')
data['Latent-Attention_Embedding'] = None

# 1. Load a pretrained Sentence Transformer model
print("Loading Sentence Transformer model...")
model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
model.max_seq_length = 32768
model.tokenizer.padding_side="right"
print("Model loaded successfully.")

def feature_extraction(sentence):
    sentence_eos = sentence + model.tokenizer.eos_token
    embedding = model.encode(sentence_eos, batch_size=1, normalize_embeddings=True)
    return embedding

def process_sentence(snippet):
    sentence = snippet['words']
    if not isinstance(sentence, str) or sentence.strip() == "":
        return None  # Return None for invalid or empty sentences

    features = feature_extraction(sentence)
    
    return {
        'features': features
    }

def parallel_processing(grouped_data):
    current_conv = None
    global final_output_path

    for index, row in tqdm(grouped_data.iterrows(), total=len(grouped_data)):
        if current_conv is None:
            current_conv = row['conversation_id']
        if current_conv != row['conversation_id']:
            print(f"Saved conversation {current_conv}")
            # Save the intermediate data to a pickle file
            with open(final_output_path, 'wb') as f:
                pickle.dump(data, f)
            current_conv = row['conversation_id']
            print(f"Processing conversation {current_conv}")

        result = process_sentence(row)
        if result:
            data.at[index, 'Latent-Attention_Embedding'] = result['features']
        else:
            data.at[index, 'Latent-Attention_Embedding'] = None

final_output_path = "/mounts/Users/cisintern/pfromm/{collection}_nv-embed_processed_output.pkl".format(collection=collection_name)
print(f"Starting parallel processing...")
parallel_processing(data)

# Save the final data to a pickle file
with open(final_output_path, 'wb') as f:
    pickle.dump(data, f)

print(f"Final output saved to {final_output_path}")
