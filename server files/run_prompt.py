import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys


def load_data(input_path):
    print(f"Loading data from {input_path}")
    df = pd.read_pickle(input_path)
    return df

def classify_text_llama_batch(model, tokenizer, texts):
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
    )

    with torch.no_grad():
        outputs = model.generate(
            batched_inputs['input_ids'].cuda(),
            attention_mask=batched_inputs['attention_mask'],
            max_new_tokens=4096,
            pad_token_id=tokenizer.eos_token_id
        )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return responses


def add_phatic_classification(df, model, tokenizer, batch_size=8):
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

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("No path provided. Using default paths.")
        output_path = "/nfs/gdata/feichin/projects/nv_embed/data_llama70B_processed_output.pkl"
        input_path = "/nfs/gdata/feichin/projects/nv_embed/data_nv-embed_processed_output.pkl"
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    
    print(f"Input path: {input_path}")

    print("Loading model...")
    model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
        )
    # model = torch.nn.DataParallel(model).cuda()
    print("Done")

    print("Loading data...")
    df = load_data(input_path)
    print("Done")

    print("Classifying...")
    df = add_phatic_classification(df, model, tokenizer)
    print("Done")

    print("Saving...")
    df.to_pickle(output_path)
    print("Done")

    print(f"Final DataFrame saved to {output_path}")