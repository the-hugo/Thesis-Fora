# adherence_mapping.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

def compute_adherence(conversations, guides, threshold=0.4):
    """
    Computes similarity scores between every facilitator's turn and every guide segment.
    Only candidate pairs with similarity above the specified threshold are returned.
    """
    results = []

    # Compute speaker turn indices (each conversation has one facilitator)
    conversations["SpeakerTurn"] = conversations.groupby(["conversation_id", "speaker_name"]).cumcount() + 1

    # Process each conversation (by collection_id)
    for collection_id, conversation_group in conversations.groupby("collection_id"):
        # Get the corresponding guide for this collection
        guide_group = guides[guides["Collection"] == collection_id]
        if guide_group.empty:
            print(f"No guide found for collection_id: {collection_id}")
            continue

        # Stack guide embeddings and extract corresponding segments and texts
        guide_embeddings = np.vstack(guide_group["Latent-Attention_Embedding"].values)
        guide_segments = guide_group["Segment"].values
        guide_texts = guide_group["Text"].values

        # Iterate over facilitator turns in the conversation
        for _, convo_row in conversation_group.iterrows():
            # Extract turn info and reshape the embedding for similarity computation
            turn_embedding = convo_row["Latent-Attention_Embedding"].reshape(1, -1)
            speaker = convo_row["speaker_name"]
            turn_text = convo_row["words"]
            speaker_turn = convo_row["SpeakerTurn"]
            conversation_id = convo_row["conversation_id"]

            # Compute cosine similarity between the turn and all guide embeddings
            similarity_scores = cosine_similarity(turn_embedding, guide_embeddings)[0]

            # Retain only those guide segments where similarity > threshold (0.4)
            for idx, score in enumerate(similarity_scores):
                if score > threshold:
                    results.append({
                        "collection_id": collection_id,
                        "speaker_name": speaker,
                        "conversation_id": conversation_id,
                        "SpeakerTurn": speaker_turn,
                        "turn_text": turn_text,
                        "guide_segment": guide_segments[idx],
                        "guide_text": guide_texts[idx],
                        "similarity_score": score,
                    })

    return pd.DataFrame(results)

def prepare_conversational_data(df):
    """
    Prepares the conversational data by ensuring embeddings exist and filtering for facilitators.
    """
    df = df.dropna(subset=["Latent-Attention_Embedding"])
    df = df[df["is_fac"] == True]
    return df

# (Optionally, you may update or remove other functions that no longer apply.)

if __name__ == "__main__":
    # Load guide data
    guide_input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\embeddings\conversational_guides_nv-embed_processed_output.pkl"
    print("Loading guide data...")
    guide_df = pd.read_pickle(guide_input_path)

    # Load conversation data and filter for facilitators
    conversations_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\embeddings\data_nv-embed_processed_output.pkl"
    print("Loading conversation data...")
    conversations = pd.read_pickle(conversations_path)
    conversations = prepare_conversational_data(conversations)

    # Compute adherence mappings (only retaining pairs above 0.4 similarity)
    print("Computing adherence mappings (similarity threshold = 0.4)...")
    adherence_results = compute_adherence(conversations, guide_df, threshold=0.4)
    
    # Save the mapping so the LLM classification can work on these candidates
    output_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\adherence_results_threshold_0.4.csv"
    adherence_results.to_csv(output_path, index=False)
    print(f"Adherence mapping results saved to {output_path}")
