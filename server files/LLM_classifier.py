import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

import pandas as pd
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

token = "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"
login(token)

def load_data(input_path):
    print(f"Loading data from {input_path}")
    df = pd.read_pickle(input_path)
    return df

model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def classify_text_llama_batch(model, tokenizer, texts, device="cuda"):
    batched_inputs = tokenizer(
        [
            f"Task: Analyze the given text snippet and classify each sentence as phatic or non-phatic.\n"
            "Calculate the phaticity score as a continuous value between 0 and 1.\n"
            f"Snippet: {text}\n"
            "Phatic Sentences Count: x\n"
            "Non-Phatic Sentences Count: y\n"
            "Total Sentences: x + y = z\n"
            "Phatic Ratio Calculation: x / z\n"
            for text in texts
        ],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            batched_inputs['input_ids'].cuda(),
            attention_mask=batched_inputs['attention_mask'].cuda(),
            max_new_tokens=4096,
            pad_token_id=tokenizer.eos_token_id
        )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    phatic_ratios = []
    ratio_prefix = "Phatic Ratio Calculation"
    for response in responses:
        if ratio_prefix in response:
            answer = response.split(ratio_prefix)[-1].strip()
            print(answer)
            phatic_ratios.append(answer)
        else:
            phatic_ratios.append("Ratio not found in response")
    
    return phatic_ratios

def add_phatic_classification(df, model, tokenizer, batch_size=16):
    total_rows = len(df)
    for start_idx in tqdm(range(0, total_rows, batch_size)):
        end_idx = min(start_idx + batch_size, total_rows)
        texts = df.loc[start_idx:end_idx, "words"].tolist()
        
        phatic_results = classify_text_llama_batch(model, tokenizer, texts)
        df.loc[start_idx:end_idx, "phatic speech"] = phatic_results

        if start_idx % (batch_size * 100) == 0 and start_idx != 0:
            temp_output_path = f"{output_path}_temp_{start_idx}.pkl"
            df.to_pickle(temp_output_path)
            print(f"Temporary DataFrame saved to {temp_output_path}")

    return df

input_path = "/mounts/Users/cisintern/pfromm/data_nv-embed_processed_output.pkl"
output_path = "/mounts/Users/cisintern/pfromm/data_llama70B_processed_output.pkl"

df = load_data(input_path)

df = add_phatic_classification(df, model, tokenizer)
df.to_pickle(output_path)
print(f"Final DataFrame saved to {output_path}")
