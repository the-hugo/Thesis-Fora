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


def classify_adherence_to_guide(df, model, tokenizer):
    """
    For each candidate pair (facilitator turn and guide snippet) in df,
    this function prompts the LLM to rate the adherence on a scale from 0 to 1.
    The LLM is expected to output only a numeric score.
    """
    # Add a new column for the LLM score if it doesn't exist yet.
    if "llm_adherence_score" not in df.columns:
        df["llm_adherence_score"] = 0.0

    pbar = tqdm(total=len(df), desc="Classifying Adherence to Guide")

    # Iterate row by row
    for idx, row in df.iterrows():
        turn_text = row["turn_text"]
        guide_text = row["guide_text"]

        # Construct a prompt that asks the LLM to rate adherence.
        input_text_str = f"""Below you have a facilitator's turn from a conversation and a corresponding guide snippet.
Facilitator Turn: "{turn_text}"
Guide Snippet: "{guide_text}"
On a scale from 0 to 1, where 1 means the turn fully adheres to the guide snippet and 0 means it does not adhere at all, please provide only a numeric score.
Answer:"""

        inputs = tokenizer(
            [input_text_str],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=16,
                pad_token_id=tokenizer.eos_token_id,
            )

        output_sequences = output[:, inputs["input_ids"].shape[-1]:]  # Exclude prompt tokens
        response = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        response_text = response[0].strip()

        # Attempt to extract a float value from the LLM response
        try:
            score = float(response_text.split()[0])
        except Exception as e:
            score = 0.0  # default in case of an error
            print(f"Error parsing LLM response at index {idx}: '{response_text}' -> {e}")

        df.at[idx, "llm_adherence_score"] = score
        pbar.update(1)
    pbar.close()
    return df


def filter_by_llm_score(df):
    """
    For each guide snippet (per conversation), keep only the candidate with the highest LLM adherence score.
    """
    # Group by conversation and guide snippet (or guide_segment) and take the row with maximum LLM score.
    filtered_df = df.loc[df.groupby(["conversation_id", "guide_segment"])["llm_adherence_score"].idxmax()]
    return filtered_df


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
    # Paths can be adjusted as needed.
    # (For adherence classification, we expect the CSV produced by the first script.)
    adherence_input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\adherence_results_threshold_0.4.csv"
    print(f"Loading adherence mapping data from: {adherence_input_path}")
    adherence_df = pd.read_csv(adherence_input_path)

    print("Loading Model 1 (SmolLM)...")
    checkpoint = "HuggingFaceTB/SmolLM-1.7B-Instruct"
    device = "cpu"  # Change to "cuda" if available.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    print("Model 1 loaded.")

    # --- New: Classify adherence to guide ---
    print("Classifying adherence to guide with LLM...")
    adherence_df = classify_adherence_to_guide(adherence_df, model, tokenizer)
    print("Filtering candidates to keep only the highest LLM score per conversation and guide segment...")
    adherence_df_filtered = filter_by_llm_score(adherence_df)
    adherence_output_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\adherence_results_classified.csv"
    adherence_df_filtered.to_csv(adherence_output_path, index=False)
    print(f"Classified adherence results saved to {adherence_output_path}")

    # (The rest of your original script—for example, classifying speech acts or personal sharing—can remain as is.)
    print("Classifying speech acts...")
    # Assuming you want to process facilitators for speech acts:
    data_input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\embeddings\data_nv-embed_processed_output.pkl"
    df = load_data(data_input_path)
    _, facilitators_df = preprocess_data(df)
    facilitators_df = add_classification(facilitators_df, model, tokenizer, "speech acts")
    # Save or further process facilitators_df as needed.
    print("Speech acts classification done.")

    print("All tasks completed.")