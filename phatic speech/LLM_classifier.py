from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import numpy as np
import json
import os
from huggingface_hub import login

login("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")

def load_data(input_path):
    print(f"Loading data from {input_path}")
    df = pd.read_pickle(input_path)
    df['Latent-Attention_Embedding'] = df['Latent-Attention_Embedding'].apply(np.array)
    initial_count = len(df)
    df = df.dropna(subset=['Latent-Attention_Embedding'])
    dropped_count = initial_count - len(df)
    print(f"Dropped {dropped_count} rows due to NaN values in 'Latent-Attention_Embedding'")
    return df

def classify_text_llama(model, tokenizer, text):
    inputs = tokenizer(
        f"Task: Analyze the given text snippet and classify each sentence as phatic or non-phatic."
        "Calculate the phaticity score as a continuous value between 0 and 1."
        "Snippet: {text}"
        "Phatic Sentences Count: x"
        "Non-Phatic Sentences Count: y"
        "Total Sentences: x + y = z"
        "Phatic Ratio Calculation: x / z",
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1000)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(response)
    ratio_prefix = "Phatic Ratio Calculation: "
    if ratio_prefix in response:
        answer = response.split(ratio_prefix)[-1].strip()
        print(answer)
        return answer
    else:
        return "Ratio not found in response"

def add_phatic_classification(df, model, tokenizer):
    df["phatic speech"] = df["words"].apply(
        lambda text: classify_text_llama(model, tokenizer, text)
    )
    return df

# Load configuration
config_path = "./config.json"
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, config_path)

with open(config_path, "r") as config_file:
    config = json.load(config_file)

input_path = config["input_path_template"]
output_path = config["output_path_template"]

# Load model and tokenizer
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load data
df = load_data(input_path)

df = add_phatic_classification(df, model, tokenizer)

df.to_pickle(output_path)
print(f"Final DataFrame saved to {output_path}")
