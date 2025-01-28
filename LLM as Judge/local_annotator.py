# script_local.py

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys


def load_data(input_path):
    print(f"Loading data from {input_path}")
    df = pd.read_pickle(input_path)
    return df


def classify_personal_sharing(df, model, tokenizer):
    """
    This function processes a DataFrame to classify personal sharing in conversation turns.
    Using only Model 1 here.
    """
    results = []
    grouped = df.groupby("conversation_id")

    for conversation_id, turns in grouped:
        turns = turns.sort_values(by="SpeakerTurn").reset_index(drop=True)

        for i in range(2, len(turns)):  # Start at index 2 (SpeakerTurn = 3)
            target_turn = turns.iloc[i]

            # Basic checks
            if target_turn["words"] is None:
                continue
            if target_turn["SpeakerTurn"] == turns["SpeakerTurn"].max():
                break
            word_count = len(target_turn["words"].split())
            if word_count < 5:
                continue

            context_turn_1 = turns.iloc[i - 2]  # Second preceding turn
            context_turn_2 = turns.iloc[i - 1]  # Immediately preceding turn

            # Prepare the prompt
            input_text_str = f"""You will be presented with text in a TARGET TURN from a speaker turn from a transcribed spoken conversation. The text was spoken by a participant in a conversation. We are identifying instances of personal sharing by the speaker. Your job is to identify the following sharing types in the quote:
 - Personal story - Personal experience
These are types of sharing that are only sometimes used. Many of the quotes will not contain either, and some will contain both. The definitions are important to make sure they actually apply.

Definitions: Personal story: A personal story describes a discrete sequence of events involving the speaker that occurred at a discrete point in time in the past or in the ongoing active present.
Personal experience: The mention of personal experience includes introductions, facts about self, professional background, general statements about the speaker’s life that are not a sequence of specific events, and repeated habitual occurrences or general experiences that are not discrete.

Here are some examples and expected correct answers for each:
Example 1 CONTEXT TURN: I’m ready to go next. CONTEXT TURN: Ok, go ahead. TARGET TURN: I started in vegetables. And then when I came to Maine, and I looked around, and I said, "Jesus, every year there’s like 12 more vegetable farms, but I can only go one place to get a steak or some sausage. Wait a minute here. There’s a niche." And that’s what I think they have to be open to. So I switched completely. Answer: Personal story
Example 2 CONTEXT TURN: You know what’s crazy? CONTEXT TURN: Go on. TARGET TURN: In my 15 years of teaching, I rarely saw students behave badly to each other. Answer: Personal experience
Example 3 CONTEXT TURN: I’m ready to go next. CONTEXT TURN: Ok, go ahead. TARGET TURN: My favorite thing is that I grew up in a town with 16,000 people so this way bigger than anything I’ve ever lived in. Unfortunately for me I’ve only lived here for like a year and a half, almost two years. So I moved right fresh out of college and into a pandemic basically, and so I guess I haven’t seen all of the great things in Madison does, but this is the biggest, and I felt very welcomed in this area so that’s that. Answer: Personal story, Personal experience
Example 4 CONTEXT TURN: Wendy, you’re on mute now. CONTEXT TURN: Ok, sorry, but it’s your turn. TARGET TURN: No, I thought it was your turn? Sorry, I’m confused. Answer: None

CONTEXT TURN: {context_turn_1["words"]}
CONTEXT TURN: {context_turn_2["words"]}
TARGET TURN: {target_turn["words"]}
Annotate the TARGET TURN for the above personal sharing types in conversation.
Do not output any explanation, just output a comma-separated list of any that apply.
If none apply, output "None". Answer:
"""

            # Tokenize
            inputs = tokenizer(
                [input_text_str],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            with torch.no_grad():
                # Move inputs to the same device as the model
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                output = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=128,  # shortened for local usage
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.batch_decode(output, skip_special_tokens=True)
            # You can parse `response` further if needed. For now, just store it:
            results.append((conversation_id, target_turn["SpeakerTurn"], response[0]))
            print(response[0])
            print(target_turn["SpeakerTurn"])
    return results


def classify_speech_acts(df, model, tokenizer):
    # Stub for your speech-act classification (unused in this example).
    pass


def add_classification(df, model, tokenizer, var):
    """
    var can be: "speech acts", "personal_sharing", "adherence_to_guide"
    """
    if var == "speech acts":
        classify_speech_acts(df, model, tokenizer)
    if var == "personal_sharing":
        classify_personal_sharing(df, model, tokenizer)
    if var == "adherence_to_guide":
        classify_personal_sharing(df, model, tokenizer)  # or another function

    return df


def preprocess_data(df):
    participants_df = df[df["is_fac"] == False]
    facilitators_df = df[df["is_fac"] == True]
    return participants_df, facilitators_df


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Example argument parsing or manual paths (adjust to your local paths)
    # -------------------------------------------------------------------------
    # if len(sys.argv) == 1:
    #     print("No path provided. Using default paths.")
    #     output_path = "/path/to/output.pkl"
    #     input_path = "/path/to/data_nv-embed_processed_output.pkl"
    # else:
    #     input_path = sys.argv[1]
    #     output_path = sys.argv[2]

    # For demonstration, setting local path:
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\embeddings\data_nv-embed_processed_output.pkl"
    print(f"Input path: {input_path}")

    # -------------------------------------------------------------------------
    # Load Model 1 (SmolLM) - CPU by default
    # -------------------------------------------------------------------------
    print("Loading Model 1 (SmolLM)...")
    checkpoint = "HuggingFaceTB/SmolLM-1.7B-Instruct"
    device = "cpu"  # or "cuda" if you want GPU locally
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    print("Model 1 loaded.")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("Loading data...")
    df = load_data(input_path)
    print("Done loading data.")

    # -------------------------------------------------------------------------
    # Preprocess
    # -------------------------------------------------------------------------
    print("Preprocessing...")
    participants_df, facilitators_df = preprocess_data(df)
    print("Done preprocessing.")

    # -------------------------------------------------------------------------
    # Classify personal sharing (example usage)
    # -------------------------------------------------------------------------
    print("Classifying personal sharing...")
    df = add_classification(df, model, tokenizer, "personal_sharing")
    print("Done")

    # -------------------------------------------------------------------------
    # Classify adherence_to_guide (example usage)
    # -------------------------------------------------------------------------
    print("Classifying adherence_to_guide...")
    df = add_classification(facilitators_df, model, tokenizer, "adherence_to_guide")
    print("Done")

    # -------------------------------------------------------------------------
    # Classify speech acts (example usage)
    # -------------------------------------------------------------------------
    print("Classifying speech acts...")
    df = add_classification(facilitators_df, model, tokenizer, "speech acts")
    print("Done")

    # -------------------------------------------------------------------------
    # Save results (commented out for demonstration)
    # -------------------------------------------------------------------------
    # df.to_pickle(output_path)
    # print(f"Final DataFrame saved to {output_path}")

    print("All tasks completed.")
