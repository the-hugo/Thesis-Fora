# script_2models.py

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor


def load_data(input_path):
    print(f"Loading data from {input_path}")
    df = pd.read_pickle(input_path)
    return df


def classify_personal_sharing(df, model, tokenizer):
    """
    This function processes a DataFrame to classify personal sharing in conversation turns.
    This uses Model 1 (SmolLM).
    """
    results = []
    grouped = df.groupby("conversation_id")

    for conversation_id, turns in grouped:
        turns = turns.sort_values(by="SpeakerTurn").reset_index(drop=True)

        for i in range(2, len(turns)):  # Start at index 2 (SpeakerTurn = 3)
            target_turn = turns.iloc[i]

            if target_turn["words"] is None:
                continue
            if target_turn["SpeakerTurn"] == turns["SpeakerTurn"].max():
                break
            word_count = len(target_turn["words"].split())
            if word_count < 5:
                continue

            context_turn_1 = turns.iloc[i - 2]
            context_turn_2 = turns.iloc[i - 1]

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

            inputs = tokenizer(
                [input_text_str],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            with torch.no_grad():
                # Move inputs to the same device as Model 1
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                output = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.batch_decode(output, skip_special_tokens=True)
            results.append((conversation_id, target_turn["SpeakerTurn"], response[0]))

    return results


def classify_speech_acts(df, pipeline_2):
    """
    Example function that uses Model 2 (phi-4) via a pipeline to do something.
    In a real scenario, you’d adapt the prompt or approach to your actual classification needs.
    """
    results = []
    grouped = df.groupby("conversation_id")

    for conversation_id, turns in grouped:
        # Minimal usage example: take the last user turn and generate something
        # (You should tailor this logic to your actual speech-acts classification.)
        if len(turns) == 0:
            continue

        # For demonstration: we pass a short prompt to pipeline_2
        example_message = [
            {"role": "system", "content": "You are a medieval knight analyzing modern conversation."},
            {"role": "user", "content": f"Please classify the speech acts in: '{turns.iloc[-1]['words']}'."}
        ]
        output = pipeline_2(example_message, max_new_tokens=64)
        # The pipeline returns a list of dicts. We'll just keep the text here:
        gen_text = output[0]["generated_text"]
        results.append((conversation_id, gen_text))

    return results


def add_classification(df, model_1, tokenizer_1, pipeline_2, var):
    """
    Modified so that it can accept both Model 1 and pipeline_2.
    """
    if var == "speech acts":
        classify_speech_acts(df, pipeline_2)
    elif var == "personal_sharing":
        classify_personal_sharing(df, model_1, tokenizer_1)
    elif var == "adherence_to_guide":
        # Just re-using the personal_sharing function as a placeholder
        classify_personal_sharing(df, model_1, tokenizer_1)

    return df


def preprocess_data(df):
    participants_df = df[df["is_fac"] == False]
    facilitators_df = df[df["is_fac"] == True]
    return participants_df, facilitators_df


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Example argument parsing or manual paths
    # -------------------------------------------------------------------------
    # if len(sys.argv) == 1:
    #     print("No path provided. Using default paths.")
    #     output_path = "/path/to/output.pkl"
    #     input_path = "/path/to/data_nv-embed_processed_output.pkl"
    # else:
    #     input_path = sys.argv[1]
    #     output_path = sys.argv[2]

    # For demonstration, set local path:
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\embeddings\data_nv-embed_processed_output.pkl"
    print(f"Input path: {input_path}")

    # -------------------------------------------------------------------------
    # Load Both Models with GPU usage (device_map="auto")
    # -------------------------------------------------------------------------
    print("Loading Model 1 (SmolLM) for classification tasks...")
    model_name_1 = "HuggingFaceTB/SmolLM-1.7B-Instruct"
    tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)

    # Set up Model 1 for GPU usage
    # If you have multiple GPUs and want them automatically used, set device_map="auto"
    model_1 = AutoModelForCausalLM.from_pretrained(
        model_name_1,
        device_map="auto",  # or remove if you want a single-GPU environment
        torch_dtype=torch.float16  # or "auto"
    )

    print("Model 1 loaded on GPU(s).")

    print("Loading Model 2 (phi-4) pipeline...")
    pipeline_2 = pipeline(
        "text-generation",
        model="microsoft/phi-4",
        model_kwargs={
            "torch_dtype": "auto",
        },
        device_map="auto"
    )
    print("Model 2 pipeline loaded on GPU(s).")

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
    # Demonstration of parallel classification
    # -------------------------------------------------------------------------
    print("Starting parallel classification with both models...")

    # We'll perform three classification tasks in parallel:
    # 1) personal_sharing on participants_df (uses Model 1)
    # 2) adherence_to_guide on facilitators_df (uses Model 1)
    # 3) speech_acts on facilitators_df (uses Model 2)

    def task_add_classification(dataframe, var):
        return add_classification(dataframe, model_1, tokenizer_1, pipeline_2, var)

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_personal = executor.submit(task_add_classification, participants_df, "personal_sharing")
        future_adherence = executor.submit(task_add_classification, facilitators_df, "adherence_to_guide")
        future_speech = executor.submit(task_add_classification, facilitators_df, "speech acts")

        # Gather results:
        result_personal = future_personal.result()
        result_adherence = future_adherence.result()
        result_speech = future_speech.result()

    print("Done parallel classification.")

    # -------------------------------------------------------------------------
    # (Optional) Combine results back to df or store individually
    # Here we simply print a message
    # -------------------------------------------------------------------------
    print("Classification tasks completed with both models in parallel.")

    # -------------------------------------------------------------------------
    # Save results (commented out for demonstration)
    # -------------------------------------------------------------------------
    # df.to_pickle(output_path)
    # print(f"Final DataFrame saved to {output_path}")

    print("All tasks completed.")
