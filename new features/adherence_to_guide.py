import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re


def compute_adherence(conversations, guides):
    """
    Computes similarity scores between every facilitator's turn and every guide segment.
    Returns a DataFrame with all matches and their respective similarity scores.
    """
    results = []

    # count the facilitator turns (words column) per conversation per facilitator
    conversations["SpeakerTurn"] = conversations.groupby(["conversation_id", "speaker_name"]).cumcount() + 1
    # now I need the max number of turns per conversation per facilitator
    max_turns = conversations.groupby(["conversation_id", "speaker_name"])["SpeakerTurn"].max().reset_index()
    # Process each conversation
    for collection_id, conversation_group in conversations.groupby("collection_id"):
        # Get the corresponding guide for this collection
        guide_group = guides[guides["Collection"] == collection_id]

        if guide_group.empty:
            print(f"No guide found for collection_id: {collection_id}")
            continue

        guide_embeddings = np.vstack(guide_group["Latent-Attention_Embedding"].values)
        guide_segments = guide_group["Segment"].values
        guide_texts = guide_group["Text"].values

        # Iterate through each facilitator's turn
        for _, convo_row in conversation_group.iterrows():
            turn_embedding = convo_row["Latent-Attention_Embedding"].reshape(1, -1)
            speaker = convo_row["speaker_name"]
            turn_text = convo_row["words"]
            speaker_turn = convo_row["SpeakerTurn"]
            conversation_id = convo_row["conversation_id"]
            # Compute similarity with all guide segments
            similarity_scores = cosine_similarity(turn_embedding, guide_embeddings)[0]

            # Store results for each segment
            for idx, score in enumerate(similarity_scores):
                results.append(
                    {
                        "collection_id": collection_id,
                        "speaker_name": speaker,
                        "turn_text": turn_text,
                        "guide_segment": guide_segments[idx],
                        "guide_text": guide_texts[idx],
                        "similarity_score": score,
                        "speaker_turn": speaker_turn,
                        "conversation_id": conversation_id,
                    }
                )

    return pd.DataFrame(results)


def prepare_conversational_data(df):
    """
    Prepares the conversational data by cleaning and filtering for facilitators.
    """
    df = df.dropna(subset=["Latent-Attention_Embedding"])
    """
    df["speaker_name"] = df["speaker_name"].str.lower().str.strip()
    df["speaker_name"] = df["speaker_name"].str.replace(r"\[.*\]", "", regex=True)
    df = df[
        ~df["speaker_name"].str.contains(
            "^speaker|moderator|audio|computer|computer voice|facilitator|group|highlight|interpreter|interviewer|multiple voices|other speaker|participant|redacted|speaker x|unknown|video"
        )
    ]
    df["speaker_name"] = df["speaker_name"].apply(lambda x: re.sub(r"^\s+|\s+$", "", x))
    """
    df = df[df["is_fac"] == True]
    return df


def filter_highest_similarity(adherence_results):
    """
    Retains only the turn_text with the highest similarity score per speaker per conversation.
    """
    # Group by collection_id and speaker_name, and keep the row with max similarity_score
    adherence_results["guide_text_order"] = adherence_results.groupby(
        ["collection_id", "speaker_name"]
    ).cumcount()
    filtered_results = adherence_results.loc[
        adherence_results.groupby(["collection_id", "speaker_name", "guide_text"])[
            "similarity_score"
        ].idxmax()
    ]
    filtered_results = filtered_results.sort_values(
        by=["collection_id", "speaker_name", "guide_text_order"]
    ).drop(columns=["guide_text_order"])
    return filtered_results


def calculate_similarity_metrics_per_conversation(mapping_df, guides):
    """
    Calculate average similarity per conversation, considering only values above 0.6,
    and also calculate the proportion of turns with similarity > 0.6.

    Args:
        mapping_df (pd.DataFrame): DataFrame containing guide-to-turn mapping with similarity scores.
        guides (pd.DataFrame): DataFrame containing the guide data.

    Returns:
        pd.DataFrame: Aggregated metrics per conversation.
    """
    
    # Add a masked similarity column where scores below 0.6 are set to 0
    mapping_df["masked_similarity"] = mapping_df["similarity_score"].apply(
        lambda x: x if x > 0.6 else 0
    )

    # Count total guide texts per collection (ensures varying guide lengths are respected)
    guide_counts = guides.groupby("Collection")["Text"].count()

    # Aggregate metrics per conversation
    metrics = (
        mapping_df.groupby(["collection_id", "conversation_id", "speaker_name"])
        .agg(
            average_similarity=(
                "masked_similarity",
                "mean",
            ),  # Average similarity (masked)
            total_turns_above_0_6=(
                "similarity_score",
                lambda x: (x > 0.6).sum(),
            ),  # Count of turns > 0.6
        )
        .reset_index()
    )

    # Normalize total guide texts based on actual guide count per collection
    metrics["total_guide_texts"] = metrics["collection_id"].map(guide_counts)

    # Add proportion of turns > 0.6
    metrics["proportion_above_0_6"] = (
        metrics["total_turns_above_0_6"] / metrics["total_guide_texts"]
    )

    return metrics


if __name__ == "__main__":
    # Load data
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\embeddings\conversational_guides_nv-embed_processed_output.pkl"
    print("Loading data")
    guide_df = pd.read_pickle(input_path)

    # Load conversations data
    conversations_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\embeddings\data_nv-embed_processed_output.pkl"
    print("Loading conversation data")
    conversations = pd.read_pickle(conversations_path)
    conversations = prepare_conversational_data(conversations)

    # Compute adherence for all conversations
    adherence_results = compute_adherence(conversations, guide_df)
    highest_similarity_results = filter_highest_similarity(adherence_results)
    similarity_metrics = calculate_similarity_metrics_per_conversation(
        highest_similarity_results, guide_df
    )
    
    similarity_metrics.rename(columns={"average_similarity": "adherence_to_guide"}, inplace=True)

    # Save results
    output_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\adherence_results_all_conversations.csv"
    highest_similarity_results.to_csv(output_path, index=False)
    print(f"Adherence results for all conversations saved to {output_path}")

    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\facilitators_features_big.csv"
    to_merge = pd.read_csv(input_path)
    to_merge = to_merge.merge(
        similarity_metrics[["conversation_id", "speaker_name", "adherence_to_guide"]],
        on=["conversation_id", "speaker_name"],
        how="left"
    )
    to_merge.to_csv(input_path, index=False)