# from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
# from transformers import logging, AutoModel
import os
# import huggingface_hub.utils

# logging.set_verbosity_error()

# huggingface_hub.utils._http.default_timeout = 230
collection_name = "corpus_data"
input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\input\{}\{}_transformed_data.csv".format(collection_name, collection_name)
data = pd.read_csv(r"data/corpus_resources/data.csv", sep=',')
data['Latent-Attention_Embedding'] = None

print(f"Max duration: {data['duration'].max()}")
print(f"Min duration: {data['duration'].min()}")
print(f"Mean duration: {data['duration'].mean()}")
    
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
model.max_seq_length = 32768
model.tokenizer.padding_side="right"


def feature_extraction(sentence):
  sentence_eos = sentence + model.tokenizer.eos_token
  embedding = model.encode(sentence_eos, batch_size=1, normalize_embeddings=True)
  return embedding


def process_sentence(snippet):
    sentence = snippet['Content']
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
            current_conv = row['Conversation ID']
        if current_conv != row['Conversation ID']:
            print("Saved", current_conv)
            data.to_csv(final_output_path, mode='a', header=not os.path.exists(final_output_path), index=False)
            current_conv = row['Conversation ID']
            print("Processing", current_conv)

        result = process_sentence(row)
        if result:
            data.at[index, 'Latent-Attention_Embedding'] = result['features']
        else:
            data.at[index, 'Latent-Attention_Embedding'] = None

final_output_path = "../../data/output/{collection}_nv-embed_processed_output.csv".format(collection=collection_name)
parallel_processing(data)

data.to_csv(final_output_path, index=False)
print(f"Final output saved to {final_output_path}")

