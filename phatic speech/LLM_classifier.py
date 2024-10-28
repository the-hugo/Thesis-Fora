import torch
import pandas as pd
from huggingface_hub import login
from unsloth import FastLanguageModel
from tqdm import tqdm

token = "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"
login(token)


def load_data(input_path):
    print(f"Loading data from {input_path}")
    df = pd.read_pickle(input_path)
    return df


def classify_text_llama(model, tokenizer, text, device="cuda"):
    inputs = tokenizer(
        f"Task: Analyze the given text snippet and classify each sentence as phatic or non-phatic.\n"
        "Calculate the phaticity score as a continuous value between 0 and 1.\n"
        "Snippet: {text}\n"
        "Phatic Sentences Count: x\n"
        "Non-Phatic Sentences Count: y\n"
        "Total Sentences: x + y = z\n"
        "Phatic Ratio Calculation: x / z\n",
        return_tensors="pt",
    ).to(device)

    model = model.to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1000)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    ratio_prefix = "Phatic Ratio Calculation"
    if ratio_prefix in response:
        answer = response.split(ratio_prefix)[-1].strip()
        print(answer)
        return answer
    else:
        return "Ratio not found in response"


def add_phatic_classification(df, model, tokenizer):
    for i in tqdm(range(len(df))):
        df.at[i, "phatic speech"] = classify_text_llama(
            model, tokenizer, df.at[i, "words"]
        )

        if i % 1000 == 0 and i != 0:
            temp_output_path = f"{output_path}_temp_{i}.pkl"
            df.to_pickle(temp_output_path)
            print(f"Temporary DataFrame saved to {temp_output_path}")

    return df


input_path = "/mounts/Users/cisintern/pfromm/data_nv-embed_processed_output.pkl"
output_path = "/mounts/Users/cisintern/pfromm/data_llama70B_processed_output.pkl"

load_in_4bit = True
model_name = "unsloth/Llama-3.1-Nemotron-70B-Instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name, load_in_4bit=load_in_4bit, token=token
)

df = load_data(input_path)

df = add_phatic_classification(df, model, tokenizer)

df.to_pickle(output_path)
print(f"Final DataFrame saved to {output_path}")
