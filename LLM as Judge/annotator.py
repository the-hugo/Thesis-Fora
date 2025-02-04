import os
# Set environment variable early to help with memory fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import re
import gc
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Login using your token.
token = "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"
login(token)

##########################
# Model & Batch Generator
##########################

print("Loading google/gemma-2-9b-it model...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
# Disable caching to reduce GPU memory buildup during generation.
model.config.use_cache = False
model.eval()  # Set model to evaluation mode.
print("Model loaded.")

def batch_generate(prompts, max_new_tokens, batch_size=8):
    """
    Tokenizes a list of prompts, performs batch generation, and returns the list of decoded outputs.
    Uses torch.no_grad() and empties GPU cache after each batch.
    
    If a CUDA OutOfMemoryError occurs, it falls back to processing prompts one-by-one.
    In the individual processing fallback, it will retry each prompt up to 3 times (with gc.collect())
    before skipping it.
    
    Additionally, every time an LLM produces a response, that response is printed to the console.
    """
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        # Tokenize with padding.
        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        batch_inputs = {k: v.to("cuda") for k, v in batch_inputs.items()}
        try:
            with torch.no_grad():
                outputs = model.generate(**batch_inputs, max_new_tokens=max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA OOM encountered with batch size {batch_size}. Processing prompts individually.")
            torch.cuda.empty_cache()
            gc.collect()
            outputs = []
            for prompt in batch_prompts:
                success = False
                retries = 0
                while not success and retries < 3:
                    try:
                        small_batch_inputs = tokenizer([prompt], return_tensors="pt", padding=True)
                        small_batch_inputs = {k: v.to("cuda") for k, v in small_batch_inputs.items()}
                        with torch.no_grad():
                            out = model.generate(**small_batch_inputs, max_new_tokens=max_new_tokens)
                        outputs.extend(out)
                        del small_batch_inputs, out
                        torch.cuda.empty_cache()
                        gc.collect()
                        success = True
                    except torch.cuda.OutOfMemoryError:
                        retries += 1
                        print(f"Retrying individual prompt generation due to OOM, attempt {retries}.")
                        torch.cuda.empty_cache()
                        gc.collect()
                if not success:
                    print("Skipping prompt due to repeated OOM.")
                    default_output = tokenizer("0", return_tensors="pt")["input_ids"][0]
                    outputs.append(default_output)
        batch_decoded = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        # Print each LLM response.
        for resp in batch_decoded:
            print("LLM response:", resp)
        responses.extend(batch_decoded)
        del batch_inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
    return responses

##############################
# Data Loading and Processing
##############################

def load_data(input_path):
    print(f"Loading data from {input_path}")
    df = pd.read_pickle(input_path)
    return df

def filter_by_llm_score(df):
    """
    For each guide snippet (per conversation), keep only the candidate with the highest LLM adherence score.
    """
    filtered_df = df.loc[df.groupby(["conversation_id", "guide_segment"])["llm_adherence_score"].idxmax()]
    return filtered_df

##############################################
# Classification Functions Using Batch Processing
##############################################

def classify_adherence_to_guide(df, batch_size=8, checkpoint_interval=1000):
    """
    For each candidate pair (facilitator turn and guide snippet) in df,
    uses the LLM to rate adherence on a scale from 0 to 1.
    
    The prompt includes a one-shot example. The function extracts the first numeric value
    from the model's response.
    """
    if "llm_adherence_score" not in df.columns:
        df["llm_adherence_score"] = 0.0

    checkpoint_path = "/home/ra37qax/adherence_checkpoint.csv"
    pbar = tqdm(total=len(df), desc="Classifying Adherence to Guide")
    
    batch_prompts = []
    batch_indices = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        turn_text = row["turn_text"]
        guide_text = row["guide_text"]
        prompt = (
            "Below you have a facilitator's turn from a conversation and a corresponding guide snippet.\n"
            "A facilitator turn may include the guide snippet in its entirety, partially, or not at all.\n\n"
            "For example, if the guide snippet is:\n"
            "    \"Welcome to this conversation of Cortico’s Local Voices Network. Thanks for agreeing to participate in this conversation!\"\n"
            "but the facilitator turn is:\n"
            "    \"All right. Well, welcome to this conversation of Cortico's Local Voices Network, who is also partnering with Public Narratives, as you heard from Jhmira earlier. "
            "Thanks for agreeing to participate in this conversation. I've begun recording at this time, and I need to share a little information with you before we begin our conversation\"\n"
            "this should be considered as adherence to the guide.\n\n"
            "Using a scale from 0 to 1 (where 1 means the turn adheres to the guide and 0 means it does not), please provide only a numeric score.\n\n"
            f"Facilitator Turn: \"{turn_text}\"\n"
            f"Guide Snippet: \"{guide_text}\"\n"
            "Answer:"
        )
        batch_prompts.append(prompt)
        batch_indices.append(idx)
        
        if len(batch_prompts) >= batch_size:
            responses = batch_generate(batch_prompts, max_new_tokens=16, batch_size=batch_size)
            for idx_inner, prompt_text, response in zip(batch_indices, batch_prompts, responses):
                response_text = response.strip()
                # Extract the first number from the response.
                match = re.search(r"(\d+(\.\d+)?)", response_text)
                if match:
                    score = float(match.group(1))
                else:
                    score = 0.0
                    print(f"Error parsing LLM response at index {idx_inner}: '{response_text}'")
                df.at[idx_inner, "llm_adherence_score"] = score
                pbar.update(1)
            batch_prompts = []
            batch_indices = []
            if (i + 1) % checkpoint_interval == 0:
                df.to_csv(checkpoint_path, index=False)
                print(f"Checkpoint saved at iteration {i + 1} to {checkpoint_path}")
    
    if batch_prompts:
        responses = batch_generate(batch_prompts, max_new_tokens=16, batch_size=len(batch_prompts))
        for idx_inner, prompt_text, response in zip(batch_indices, batch_prompts, responses):
            response_text = response.strip()
            match = re.search(r"(\d+(\.\d+)?)", response_text)
            if match:
                score = float(match.group(1))
            else:
                score = 0.0
                print(f"Error parsing LLM response at index {idx_inner}: '{response_text}'")
            df.at[idx_inner, "llm_adherence_score"] = score
            pbar.update(1)
    pbar.close()
    return df

def classify_personal_sharing(df, batch_size=8):
    """
    Classify personal sharing in conversation turns using the LLM.
    Returns a list of tuples: (conversation_id, SpeakerTurn, personal sharing classification response).
    """
    results_meta = []
    prompts = []
    grouped = df.groupby("conversation_id")
    
    for conversation_id, turns in grouped:
        turns = turns.sort_values(by="SpeakerTurn").reset_index(drop=True)
        for i in range(2, len(turns)):
            target_turn = turns.iloc[i]
            if target_turn["words"] is None:
                continue
            if target_turn["SpeakerTurn"] == turns["SpeakerTurn"].max():
                break
            if len(target_turn["words"].split()) < 5:
                continue

            context_turn_1 = turns.iloc[i - 2]
            context_turn_2 = turns.iloc[i - 1]
            prompt = (
                f"You will be presented with text in a TARGET TURN from a transcribed conversation. The text was spoken by a participant. "
                f"We are identifying instances of personal sharing. Identify the following sharing types in the quote:\n"
                f" - Personal story\n - Personal experience\n\n"
                f"Definitions:\n"
                f"Personal story: A personal story describes a discrete sequence of events involving the speaker that occurred at a "
                f"discrete point in time in the past or in the ongoing active present.\n"
                f"Personal experience: The mention of personal experience includes introductions, facts about self, professional background, "
                f"general statements about the speaker’s life that are not a sequence of specific events, and repeated habitual occurrences.\n\n"
                f"Examples:\n"
                f"Example 1 CONTEXT TURN: I’m ready to go next. CONTEXT TURN: Ok, go ahead. TARGET TURN: I started in vegetables. "
                f"And then when I came to Maine... Answer: Personal story\n"
                f"Example 2 CONTEXT TURN: You know what’s crazy? CONTEXT TURN: Go on. TARGET TURN: In my 15 years of teaching, I rarely saw students behave badly. "
                f"Answer: Personal experience\n"
                f"Example 3 CONTEXT TURN: I’m ready to go next. CONTEXT TURN: Ok, go ahead. TARGET TURN: My favorite thing is that I grew up in a town... "
                f"Answer: Personal story, Personal experience\n"
                f"Example 4 CONTEXT TURN: Wendy, you’re on mute now. CONTEXT TURN: Ok, sorry, but it’s your turn. TARGET TURN: No, I thought it was your turn? "
                f"Sorry, I’m confused. Answer: None\n\n"
                f"CONTEXT TURN: {context_turn_1['words']}\n"
                f"CONTEXT TURN: {context_turn_2['words']}\n"
                f"TARGET TURN: {target_turn['words']}\n"
                f"Annotate the TARGET TURN for the above personal sharing types.\n"
                f"Do not output any explanation, just output a comma-separated list of any that apply. If none apply, output \"None\".\nAnswer:"
            )
            results_meta.append((conversation_id, target_turn["SpeakerTurn"]))
            prompts.append(prompt)
    
    responses = batch_generate(prompts, max_new_tokens=128, batch_size=batch_size) if prompts else []
    final_results = []
    for (conversation_id, speaker_turn), prompt, response in zip(results_meta, prompts, responses):
        response_text = response.strip()
        final_results.append((conversation_id, speaker_turn, response_text))
    return final_results

def classify_speech_acts(df, batch_size=8):
    """
    Classify speech acts using the LLM.
    This function processes a DataFrame to classify speech acts (express appreciation,
    express affirmation, open invitation, specific invitation) in conversation turns.
    It updates the DataFrame with new columns.
    """
    # Ensure the needed columns exist before we start filling them
    for col in [
        "speech_acts_result",
        "express_appreciation",
        "express_affirmation",
        "open_invitation",
        "specific_invitation",
    ]:
        if col not in df.columns:
            df[col] = ""
    
    grouped = df.groupby("conversation_id")
    conversation_ids = list(grouped.groups.keys())
    pbar = tqdm(total=len(df), desc="Classifying Speech Acts")
    
    for conversation_id in conversation_ids:
        turns = grouped.get_group(conversation_id)
        turns = turns.sort_values(by="SpeakerTurn").reset_index(drop=True)
        
        for i in range(2, len(turns)):  # Start at index 2 (SpeakerTurn = 3)
            target_turn = turns.iloc[i]
            pbar.update(1)
            
            # Basic checks
            if target_turn["words"] is None:
                continue
            if target_turn["SpeakerTurn"] == turns["SpeakerTurn"].max():
                break
            if len(target_turn["words"].split()) < 5:
                continue
            
            context_turn_1 = turns.iloc[i - 2]  # Second preceding turn
            context_turn_2 = turns.iloc[i - 1]  # Immediately preceding turn
            
            # Prepare the prompt for speech acts
            input_text_str = f"""You will be presented with text in a TARGET TURN from a speaker turn in a transcribed spoken conversation. The text was spoken by a participant in a conversation. We are identifying the following speech acts in the quote:
 - Express appreciation
 - Express affirmation
 - Open invitation
 - Specific invitation

Definitions (derived from examples):
- Express appreciation: The speaker conveys gratitude or thanks, or positively acknowledges another’s contribution or presence.
- Express affirmation: The speaker indicates agreement, support, or understanding of another's statement.
- Open invitation: The speaker invites anyone in the group (without naming a specific individual) to speak or act.
- Specific invitation: The speaker explicitly calls on a particular individual to speak or act.

CONTEXT TURN: {context_turn_1["words"]}
CONTEXT TURN: {context_turn_2["words"]}
TARGET TURN: {target_turn["words"]}

Annotate the TARGET TURN for the above speech acts. Do not output any explanation, just output a comma-separated list of any that apply. If none apply, output "None". Answer:"""
            
            # Tokenize
            inputs = tokenizer(
                [input_text_str],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            
            with torch.no_grad():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                output = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            output_sequences = output[:, inputs["input_ids"].shape[-1]:]  # Exclude input tokens
            response = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            classification = response[0].strip()
            
            # Print the LLM response for this speech act classification.
            print(f"LLM response for conversation {conversation_id}, SpeakerTurn {target_turn['SpeakerTurn']}: {classification}")
            
            # Locate the row in df that corresponds to target_turn by conversation_id and SpeakerTurn
            idx = df.index[
                (df["conversation_id"] == conversation_id)
                & (df["SpeakerTurn"] == target_turn["SpeakerTurn"])
            ].tolist()
            
            if len(idx) == 1:
                row_idx = idx[0]
                df.at[row_idx, "speech_acts_result"] = classification
                
                # Reset columns for this row
                df.at[row_idx, "express_appreciation"] = 0
                df.at[row_idx, "express_affirmation"] = 0
                df.at[row_idx, "open_invitation"] = 0
                df.at[row_idx, "specific_invitation"] = 0
                
                # Check for each speech act in the classification string
                if "Express appreciation" in classification or "express appreciation" in classification:
                    df.at[row_idx, "express_appreciation"] = 1
                if "Express affirmation" in classification or "express affirmation" in classification:
                    df.at[row_idx, "express_affirmation"] = 1
                if "Open invitation" in classification or "open invitation" in classification:
                    df.at[row_idx, "open_invitation"] = 1
                if "Specific invitation" in classification or "specific invitation" in classification:
                    df.at[row_idx, "specific_invitation"] = 1
            
            # Save a checkpoint every 1000 updates.
            if pbar.n % 1000 == 0:
                df.to_pickle("temp_classification_checkpoint.pkl")
    
    pbar.close()
    return df

def add_classification(df, var, batch_size=8):
    """
    Depending on the classification type, runs the corresponding function.
    """
    if var == "speech acts":
        return classify_speech_acts(df, batch_size=batch_size)
    elif var == "personal_sharing":
        return classify_personal_sharing(df, batch_size=batch_size)
    elif var == "adherence_to_guide":
        return classify_adherence_to_guide(df, batch_size=batch_size)
    return df

def preprocess_data(df):
    participants_df = df[df["is_fac"] == False]
    facilitators_df = df[df["is_fac"] == True]
    return participants_df, facilitators_df

##############################################
# Main Execution
##############################################

if __name__ == "__main__":
    # --- Adherence-to-Guide Classification ---
    adherence_input_path = "/home/ra37qax/adherence_results_threshold_0.4.csv"
    print(f"Loading adherence mapping data from: {adherence_input_path}")
    adherence_df = pd.read_csv(adherence_input_path)
    
    print("Classifying adherence to guide with the new LLM...")
    adherence_df = classify_adherence_to_guide(adherence_df, batch_size=8)
    print("Filtering candidates to keep only the highest LLM score per conversation and guide segment...")
    adherence_df_filtered = filter_by_llm_score(adherence_df)
    
    # Save adherence results as CSV and PKL.
    adherence_output_csv = "/home/ra37qax/adherence_results_classified.csv"
    adherence_output_pkl = "/home/ra37qax/adherence_results_classified.pkl"
    adherence_df_filtered.to_csv(adherence_output_csv, index=False)
    adherence_df_filtered.to_pickle(adherence_output_pkl)
    print(f"Classified adherence results saved to {adherence_output_csv} and {adherence_output_pkl}")
    
    # --- Speech Acts and Personal Sharing Classification ---
    print("Classifying speech acts and personal sharing...")
    data_input_path = "/home/ra37qax/data_nv-embed_processed_output.pkl"
    df = load_data(data_input_path)
    _, facilitators_df = preprocess_data(df)
    
    # Compute speech acts classification. (Make a copy if you need to preserve the original order.)
    speech_acts_df = add_classification(facilitators_df.copy(), "speech acts", batch_size=8)
    
    # Compute personal sharing classification results (list of tuples).
    personal_sharing_results = classify_personal_sharing(facilitators_df, batch_size=8)
    # Convert list to DataFrame.
    personal_sharing_df = pd.DataFrame(personal_sharing_results, columns=["conversation_id", "SpeakerTurn", "personal_sharing_result"])
    
    # Merge speech acts and personal sharing results on conversation_id and SpeakerTurn.
    combined_df = pd.merge(speech_acts_df, personal_sharing_df, on=["conversation_id", "SpeakerTurn"], how="left")
    combined_df["personal_sharing_result"] = combined_df["personal_sharing_result"].fillna("None")
    
    # Save combined results as CSV and PKL.
    output_csv = "/home/ra37qax/speech_acts_personal_sharing_classification.csv"
    output_pkl = "/home/ra37qax/speech_acts_personal_sharing_classification.pkl"
    combined_df.to_csv(output_csv, index=False)
    combined_df.to_pickle(output_pkl)
    print(f"Speech acts and personal sharing classification results saved to {output_csv} and {output_pkl}")
    
    print("All tasks completed.")
