import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from transformers import pipeline, logging
import pandas as pd
from tqdm import tqdm

# Disable oneDNN optimizations (if needed)
logging.set_verbosity_error()

# Load the CSV data
data = pd.read_csv("./data/transformed_output.csv")

# Group by Conversation ID and Speaker ID, as we need to distinguish between conversations and speakers
grouped_data = data.groupby(['Conversation ID', 'Speaker ID', 'Index in Conversation'])

# print all Nan rows
for name, group in grouped_data:
    sentence = group['Content'].iloc[0]
    if type(sentence) != str:
        print(group)
        break