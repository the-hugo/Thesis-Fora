import os
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Set environment variable early to help with memory fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


##############################
# Setup & Model Loading
##############################


def setup_huggingface(token: str) -> None:
    """Log in to Hugging Face using the provided token."""
    login(token)


def load_model_and_tokenizer(model_name: str):
    """Load the specified model and tokenizer from Hugging Face."""
    print(f"Loading {model_name} model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    # Disable caching to reduce GPU memory buildup during generation.
    model.config.use_cache = False
    model.eval()  # Set model to evaluation mode.
    print("Model loaded.")
    return tokenizer, model


##############################
# Batch Generation Helper
##############################


def batch_generate(prompts, max_new_tokens, batch_size=4, verbose=False):
    """
    Tokenizes a list of prompts and generates model outputs in batches.
    Falls back to processing prompts one-by-one on CUDA OOM errors.
    """
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        # Tokenize with padding.
        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        batch_inputs = {k: v.to("cuda") for k, v in batch_inputs.items()}

        try:
            with torch.no_grad():
                outputs = model.generate(**batch_inputs, max_new_tokens=max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            print(
                f"CUDA OOM encountered with batch size {batch_size}. Processing prompts individually."
            )
            outputs = []
            for prompt in batch_prompts:
                success = False
                retries = 0
                while not success and retries < 3:
                    try:
                        small_batch = tokenizer(
                            [prompt], return_tensors="pt", padding=True
                        )
                        small_batch = {k: v.to("cuda") for k, v in small_batch.items()}
                        with torch.no_grad():
                            out = model.generate(
                                **small_batch, max_new_tokens=max_new_tokens
                            )
                        outputs.extend(out)
                        del small_batch, out
                        success = True
                    except torch.cuda.OutOfMemoryError:
                        retries += 1
                        print(
                            f"Retrying individual prompt generation due to OOM, attempt {retries}."
                        )
                if not success:
                    print("Skipping prompt due to repeated OOM.")
                    default_output = tokenizer("0", return_tensors="pt")["input_ids"][0]
                    outputs.append(default_output)

        batch_decoded = [
            tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]
        if verbose:
            for resp in batch_decoded:
                print("LLM response:", resp)
        responses.extend(batch_decoded)
        del batch_inputs, outputs
    return responses


def parse_numeric_response(response: str, idx) -> float:
    """
    Parses the number following 'Answer:' in the response text.
    If no such number is found, returns 0.0 and prints an error message.
    """
    response_text = response.strip()
    # print(response_text)
    match = re.search(r"Answer:\s*(\d+(?:\.\d+)?)", response_text)
    if match:
        return float(match.group(1))
    else:
        print(f"Error parsing LLM response at index {idx}: '{response_text}'")
        return 0.0


##############################
# Data Loading & Preprocessing
##############################


def load_data(input_path: str) -> pd.DataFrame:
    print(f"Loading data from {input_path}")
    return pd.read_pickle(input_path)


def filter_by_llm_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each guide snippet (per conversation), if all llm_adherence_score values are 0,
    return the row with the highest similarity_score.
    Otherwise, return the row with the highest llm_adherence_score.
    """

    def select_row(group):
        if (group["llm_adherence_score"] == 0).all():
            return group.loc[group["similarity_score"].idxmax()]
        else:
            return group.loc[group["llm_adherence_score"].idxmax()]

    filtered_df = df.groupby(["conversation_id", "guide_text"], group_keys=False).apply(
        select_row
    )
    return filtered_df


def preprocess_data(df: pd.DataFrame):
    """
    Splits the DataFrame into participants and facilitators.
    """
    participants_df = df[df["is_fac"] == False]
    facilitators_df = df[df["is_fac"] == True]
    return participants_df, facilitators_df


##############################
# Classification Functions
##############################


def classify_adherence_to_guide(
    df: pd.DataFrame, batch_size=4, checkpoint_interval=1000
) -> pd.DataFrame:
    """
    For each candidate pair in df, uses the LLM to rate adherence to the guide on a scale from 0 to 1.
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
            prompt
        ) = f"""Below you have a conversation snippet from a facilitator and a corresponding guide snippet. Your task is to decide if the facilitator's turn contains any part of the guide snippet (even if paraphrased or partially included). Only output the number 1 if it does, or 0 if it does not. Do not provide any extra text or explanation.

Example 1:
- Guide Snippet:
    "Welcome to this conversation of Cortico’s Local Voices Network. Thanks for agreeing to participate in this conversation!"
- Facilitator Turn:
    "All right. Well, welcome to this conversation of Cortico's Local Voices Network, who is also partnering with Public Narratives, as you heard from Jhmira earlier. Thanks for agreeing to participate in this conversation. I've begun recording at this time, and I need to share a little information with you before we begin our conversation."
Output: 1

Example 2:
- Guide Snippet:
    "Welcome to this conversation of Cortico’s Local Voices Network. Thanks for agreeing to participate in this conversation!"
- Facilitator Turn:
    "Today, we will focus on discussing community engagement strategies."
Output: 0

Now, evaluate the following:

Facilitator Turn: "{turn_text}"
Guide Snippet: "{guide_text}"
Answer:"""

        batch_prompts.append(prompt)
        batch_indices.append(idx)

        if len(batch_prompts) >= batch_size:
            responses = batch_generate(
                batch_prompts, max_new_tokens=16, batch_size=batch_size
            )
            for idx_inner, response in zip(batch_indices, responses):
                score = parse_numeric_response(response, idx_inner)
                df.at[idx_inner, "llm_adherence_score"] = score
                pbar.update(1)
            batch_prompts, batch_indices = [], []
            if (i + 1) % checkpoint_interval == 0:
                df.to_csv(checkpoint_path, index=False)
                print(f"Checkpoint saved at iteration {i + 1} to {checkpoint_path}")

    if batch_prompts:
        responses = batch_generate(
            batch_prompts, max_new_tokens=16, batch_size=len(batch_prompts)
        )
        for idx_inner, response in zip(batch_indices, responses):
            score = parse_numeric_response(response, idx_inner)
            df.at[idx_inner, "llm_adherence_score"] = score
            pbar.update(1)
    pbar.close()
    return df


def classify_personal_sharing(df: pd.DataFrame, batch_size=4):
    """
    Classify personal sharing in conversation turns using the LLM.
    Returns a list of tuples: (conversation_id, SpeakerTurn, personal_sharing_result).
    """
    final_results = []
    batch_prompts = []
    results_meta = []

    # Count total valid prompts for progress tracking.
    total_prompts = 0
    for conversation_id, turns in df.groupby("conversation_id"):
        turns = turns.sort_values("SpeakerTurn").reset_index(drop=True)
        for i in range(2, len(turns)):
            target_turn = turns.iloc[i]
            if (
                target_turn["words"] is None
                or target_turn["SpeakerTurn"] == turns["SpeakerTurn"].max()
                or len(target_turn["words"].split()) < 5
            ):
                continue
            total_prompts += 1

    pbar = tqdm(total=total_prompts, desc="Classifying Personal Sharing", leave=True)

    for conversation_id, turns in df.groupby("conversation_id"):
        turns = turns.sort_values("SpeakerTurn").reset_index(drop=True)
        for i in range(2, len(turns)):
            row = turns.iloc[i]
            if (
                row["words"] is None
                or row["SpeakerTurn"] == turns["SpeakerTurn"].max()
                or len(row["words"].split()) < 5
            ):
                continue
            target_turn_text = row["words"]
            context_turn_1 = turns.iloc[i - 2]["words"]
            context_turn_2 = turns.iloc[i - 1]["words"]
            prompt = (
                "You will be presented with text in a TARGET TURN from a transcribed conversation. The text was spoken by a participant. "
                "We are identifying instances of personal sharing. Identify the following sharing types in the quote:\n"
                " - Personal story\n - Personal experience\n\n"
                "Definitions:\n"
                "Personal story: A personal story describes a discrete sequence of events involving the speaker that occurred at a "
                "discrete point in time in the past or in the ongoing active present.\n"
                "Personal experience: The mention of personal experience includes introductions, facts about self, professional background, "
                "general statements about the speaker’s life that are not a sequence of specific events, and repeated habitual occurrences.\n\n"
                "Examples:\n"
                "Example 1 CONTEXT TURN: I’m ready to go next. CONTEXT TURN: Ok, go ahead. TARGET TURN: I started in vegetables. "
                "And then when I came to Maine... Answer: Personal story\n"
                "Example 2 CONTEXT TURN: You know what’s crazy? CONTEXT TURN: Go on. TARGET TURN: In my 15 years of teaching, I rarely saw students behave badly. "
                "Answer: Personal experience\n"
                "Example 3 CONTEXT TURN: I’m ready to go next. CONTEXT TURN: Ok, go ahead. TARGET TURN: My favorite thing is that I grew up in a town... "
                "Answer: Personal story, Personal experience\n"
                "Example 4 CONTEXT TURN: Wendy, you’re on mute now. CONTEXT TURN: Ok, sorry, but it’s your turn. TARGET TURN: No, I thought it was your turn? "
                "Sorry, I’m confused. Answer: None\n\n"
                f"CONTEXT TURN: {context_turn_1}\n"
                f"CONTEXT TURN: {context_turn_2}\n"
                f"TARGET TURN: {target_turn_text}\n"
                "Annotate the TARGET TURN for the above personal sharing types.\n"
                'Do not output any explanation, just output a comma-separated list of any that apply. If none apply, output "None".\nAnswer:'
            )
            # Append the conversation id and the original speaker turn (from the row)
            results_meta.append((conversation_id, row["SpeakerTurn"]))
            batch_prompts.append(prompt)

            if len(batch_prompts) >= batch_size:
                responses = batch_generate(
                    batch_prompts,
                    max_new_tokens=128,
                    batch_size=batch_size,
                    verbose=False,
                )
                for meta, response in zip(results_meta, responses):
                    response_text = response.strip()
                    if "Answer:" in response_text:
                        response_text = response_text.split("Answer:")[-1].strip()
                    final_results.append((meta[0], meta[1], response_text))
                    pbar.update(1)
                batch_prompts, results_meta = [], []

    if batch_prompts:
        responses = batch_generate(
            batch_prompts,
            max_new_tokens=128,
            batch_size=len(batch_prompts),
            verbose=False,
        )
        for meta, response in zip(results_meta, responses):
            response_text = response.strip()
            if "Answer:" in response_text:
                response_text = response_text.split("Answer:")[-1].strip()
            final_results.append((meta[0], meta[1], response_text))
            pbar.update(1)
    pbar.close()
    return final_results


def classify_speech_acts(df: pd.DataFrame, batch_size=4) -> pd.DataFrame:
    """
    Classify speech acts for facilitator turns using the LLM.
    """
    # Ensure required columns exist.
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

    # Calculate the total number of valid target turns for progress tracking.
    total_turns = 0
    for conversation_id in conversation_ids:
        turns = (
            grouped.get_group(conversation_id)
            .sort_values("SpeakerTurn")
            .reset_index(drop=True)
        )
        for i in range(2, len(turns)):
            target_turn = turns.iloc[i]
            if (
                target_turn["words"] is None
                or target_turn["SpeakerTurn"] == turns["SpeakerTurn"].max()
                or len(target_turn["words"].split()) < 5
            ):
                continue
            total_turns += 1

    pbar = tqdm(total=total_turns, desc="Classifying Speech Acts", leave=True)
    try:
        for conversation_id in conversation_ids:
            turns = (
                grouped.get_group(conversation_id)
                .sort_values("SpeakerTurn")
                .reset_index(drop=True)
            )
            for i in range(2, len(turns)):
                target_turn = turns.iloc[i]
                if (
                    target_turn["words"] is None
                    or target_turn["SpeakerTurn"] == turns["SpeakerTurn"].max()
                    or len(target_turn["words"].split()) < 5
                ):
                    continue

                target_turn_text = target_turn["words"]
                pbar.update(1)
                context_turn_1 = turns.iloc[i - 2]["words"]
                context_turn_2 = turns.iloc[i - 1]["words"]

                input_text_str = f"""You are an assistant tasked with classifying a participant’s speech in a transcribed conversation. Your goal is to identify which of the following speech acts are present in the TARGET TURN:

- **Express affirmation:** Indicates agreement, support, or understanding of another’s statement.
- **Express appreciation:** Conveys gratitude, thanks, or positively acknowledges another’s contribution.
- **Open invitation to participate:** Invites anyone in the group to speak (e.g., “Anyone else want to share a story?” or “Can I get everyone's consent?”).
- **Specific invitation to participate:** Directly invites a particular person to speak (e.g., “Gail?” or “Emily, do you want to go next?”).

Instructions:
1. Read the conversation turns provided.
2. Identify which speech act(s) apply to the TARGET TURN.
3. Output your answer as a comma-separated list of applicable speech acts (e.g., "Express affirmation, Express appreciation").
4. If no speech acts apply, simply output "None".
5. Do not include any explanation.

Examples:

**Example 1**  
CONTEXT TURN: I'm ready to go next.  
CONTEXT TURN: I thought I just gave my answer.  
TARGET TURN: Oh ok, thanks for sharing your story. I agree with what you’re saying about the city planning. I was going to encourage us to share ideas on that next.  
**Answer:** Express affirmation, Express appreciation

**Example 2**  
CONTEXT TURN: Do you mind if we move on?  
CONTEXT TURN: Ok sure, I can hold off until others share.  
TARGET TURN: Thanks. I’m trying to make sure we share time.  
**Answer:** Express appreciation

**Example 3**  
CONTEXT TURN: Ok. What do you think?  
CONTEXT TURN: Yeah. The public pools are too crowded and I wish we had a better management system.  
TARGET TURN: Definitely. I hear that. How about you, Sandy?  
**Answer:** Express affirmation

**Example 4**  
CONTEXT TURN: Wendy, you're on mute now.  
CONTEXT TURN: Ok, sorry, but it's your turn.  
TARGET TURN: No, I thought it was your turn? Sorry, I'm confused.  
**Answer:** None

**Example 5**  
CONTEXT TURN: I'm ready to go next.  
CONTEXT TURN: I thought I just gave my answer.  
TARGET TURN: Oh ok, thanks for sharing your story. I agree with what you’re saying about the city planning. I was going to encourage us to share ideas on that next. Does anyone else want to share?  
**Answer:** Open invitation to participate

**Example 6**  
CONTEXT TURN: Do you mind if we move on?  
CONTEXT TURN: Ok sure, I can hold off until others share.  
TARGET TURN: Thanks. I’m trying to make sure we share time. Sally, will you be ready to go next?  
**Answer:** Specific invitation to participate

**Example 7**  
CONTEXT TURN: Ok. What do you think?  
CONTEXT TURN: Yeah. The public pools are too crowded and I wish we had a better management system.  
TARGET TURN: Definitely. I hear that. How about you, Sandy? Or does anyone else want to go? The floor is open.  
**Answer:** Open invitation to participate, Specific invitation to participate

**Example 8**  
CONTEXT TURN: Are you able to expand more on that?  
CONTEXT TURN: Give me a second.  
TARGET TURN: Great. We will wait. Let us know when you are ready.  
**Answer:** Specific invitation to participate

Conversation for Classification:

CONTEXT TURN: {context_turn_1} 
CONTEXT TURN: {context_turn_2} 
TARGET TURN: {target_turn_text}  

**Identify the applicable speech acts in the TARGET TURN according to the definitions above.**
Output only a comma-separated list of applicable speech acts. If none apply, output "None". Do not provide any explanations.

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
                        max_new_tokens=128,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.batch_decode(output, skip_special_tokens=True)
                #print("Full LLM response:", response)

                full_response = response[0].strip()
                # Extract only the answer portion after "Answer:"
                if "Answer:" in full_response:
                    classification = full_response.split("Answer:")[-1].strip()
                else:
                    classification = full_response

                idx = df.index[
                    (df["conversation_id"] == conversation_id)
                    & (df["SpeakerTurn"] == target_turn["SpeakerTurn"])
                ].tolist()
                if len(idx) == 1:
                    row_idx = idx[0]
                    df.at[row_idx, "speech_acts_result"] = classification
                    df.at[row_idx, "express_appreciation"] = 1 if "appreciation" in classification.lower() else 0
                    df.at[row_idx, "express_affirmation"] = 1 if "affirmation" in classification.lower() else 0
                    df.at[row_idx, "open_invitation"] = 1 if "open" in classification.lower() else 0
                    df.at[row_idx, "specific_invitation"] = 1 if "specific" in classification.lower() else 0
    except Exception as e:
        print(f"Error processing conversation {conversation_id}: {e}", flush=True)
    finally:
        pbar.close()

    return df



def add_classification(df: pd.DataFrame, var: str, batch_size=4):
    """
    Runs the corresponding classification function based on the type.
    """
    if var == "speech acts":
        return classify_speech_acts(df, batch_size=batch_size)
    elif var == "personal_sharing":
        return classify_personal_sharing(df, batch_size=batch_size)
    elif var == "adherence_to_guide":
        return classify_adherence_to_guide(df, batch_size=batch_size)
    return df


##############################
# Main Execution
##############################


def main():
    # --- Setup ---
    HF_TOKEN = "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"
    setup_huggingface(HF_TOKEN)
    global tokenizer, model
    tokenizer, model = load_model_and_tokenizer("google/gemma-2-9b-it")

    # --- Adherence-to-Guide Classification ---
    adherence_input_path = "/home/ra37qax/adherence_results_threshold_0.4.csv"
    print(f"Loading adherence mapping data from: {adherence_input_path}")
    adherence_df = pd.read_csv(adherence_input_path)
    #adherence_df = adherence_df.sample(frac=0.1)
    print("Classifying adherence to guide with the new LLM...")
    adherence_df = classify_adherence_to_guide(adherence_df, batch_size=4)
    # save the results
    adherence_df.to_csv(adherence_input_path, index=False)

    print(
        "Filtering candidates to keep only the highest LLM score per conversation and guide segment..."
    )
    adherence_df_filtered = filter_by_llm_score(adherence_df)

    adherence_output_csv = "/home/ra37qax/adherence_results_classified.csv"
    adherence_output_pkl = "/home/ra37qax/adherence_results_classified.pkl"
    adherence_df_filtered.to_csv(adherence_output_csv, index=False)
    adherence_df_filtered.to_pickle(adherence_output_pkl)
    print(
        f"Classified adherence results saved to {adherence_output_csv} and {adherence_output_pkl}"
    )

    # --- Speech Acts and Personal Sharing Classification ---
    print("Classifying speech acts and personal sharing...")
    data_input_path = "/home/ra37qax/data_nv-embed_processed_output.pkl"
    df = load_data(data_input_path)
    participants_df, facilitators_df = preprocess_data(df)
    #facilitators_df = facilitators_df.sample(frac=0.1)
    # For speech acts, we classify only facilitator turns.
    speech_acts_df = add_classification(
        facilitators_df.copy(), "speech acts", batch_size=4
    )

    speech_acts_csv_path = "/home/ra37qax/speech_acts_classification.csv"
    speech_acts_pkl_path = "/home/ra37qax/speech_acts_classification.pkl"
    speech_acts_df.to_csv(speech_acts_csv_path, index=False)
    speech_acts_df.to_pickle(speech_acts_pkl_path)
    print(
        f"Speech acts classification results saved to {speech_acts_csv_path} and {speech_acts_pkl_path}"
    )
    #df = df.sample(frac=0.1)
    # Personal sharing classification on the entire dataset.
    personal_sharing_results = classify_personal_sharing(df, batch_size=4)
    personal_sharing_df = pd.DataFrame(
        personal_sharing_results,
        columns=["conversation_id", "SpeakerTurn", "personal_sharing_result"],
    )

    # --- Merge the Results ---
    df_final = df.copy()
    # Merge speech acts (facilitators) results.
    speech_acts_df = pd.read_pickle(speech_acts_pkl_path)
    df_final = pd.merge(
        df_final,
        speech_acts_df[
            [
                "conversation_id",
                "SpeakerTurn",
                "speech_acts_result",
                "express_appreciation",
                "express_affirmation",
                "open_invitation",
                "specific_invitation",
            ]
        ],
        on=["conversation_id", "SpeakerTurn"],
        how="left",
    )
    # Merge personal sharing results.
    df_final = pd.merge(
        df_final,
        personal_sharing_df,
        on=["conversation_id", "SpeakerTurn"],
        how="left",
    )
    df_final["personal_sharing_result"] = df_final["personal_sharing_result"].fillna(
        "None"
    )

    output_csv = "/home/ra37qax/speech_acts_personal_sharing_classification.csv"
    output_pkl = "/home/ra37qax/speech_acts_personal_sharing_classification.pkl"
    df_final.to_csv(output_csv, index=False)
    df_final.to_pickle(output_pkl)
    print(f"Combined classification results saved to {output_csv} and {output_pkl}")

    print("All tasks completed.")


if __name__ == "__main__":
    main()
