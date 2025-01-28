import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re

def load_data(input_path):
    """Load data from a pickle file."""
    print(f"Loading data from {input_path}")
    return pd.read_pickle(input_path)


def compute_cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    return 1 - cosine(embedding1, embedding2)


def compute_responsivity(df, window_size=6, decay_rate=0.1):
    """
    Compute speaker responsivity matrix and time-varying responsivity.

    Args:
        df (pd.DataFrame): DataFrame containing conversation data.
        window_size (int): Time window for considering previous turns (in seconds).
        decay_rate (float): Exponential decay rate for weighting past turns.

    Returns:
        pd.DataFrame: Responsivity matrix.
        dict: Time-varying responsivity signals.
    """
    speaker_names = df["speaker_name"].unique()
    num_speakers = len(speaker_names)
    embeddings = df["Latent-Attention_Embedding"].tolist()

    speaker_idx = {name: i for i, name in enumerate(speaker_names)}

    responsivity_matrix = np.zeros((num_speakers, num_speakers))
    time_varying_responsivity = {speaker: [] for speaker in speaker_names}

    for i, speaker_A in enumerate(speaker_names):
        for j, speaker_B in enumerate(speaker_names):
            if speaker_A == speaker_B:
                continue
            speaker_A_turns = df[df["speaker_name"] == speaker_A]
            speaker_B_turns = df[df["speaker_name"] == speaker_B]

            similarities = []
            for _, turn_B in speaker_B_turns.iterrows():
                embedding_B = turn_B["Latent-Attention_Embedding"]
                timestamp_B = turn_B["SpeakerTurn"]

                relevant_A_turns = speaker_A_turns[
                    (speaker_A_turns["SpeakerTurn"] < timestamp_B)
                    & (speaker_A_turns["SpeakerTurn"] >= timestamp_B - window_size)
                ]

                if relevant_A_turns.empty:
                    continue

                for _, turn_A in relevant_A_turns.iterrows():
                    embedding_A = turn_A["Latent-Attention_Embedding"]
                    timestamp_A = turn_A["SpeakerTurn"]
                    time_diff = timestamp_B - timestamp_A

                    weight = math.exp(-decay_rate * time_diff)
                    similarity = compute_cosine_similarity(embedding_A, embedding_B)
                    similarities.append(weight * similarity)

            if similarities:
                responsivity_matrix[i, j] = np.mean(similarities)

            time_varying_responsivity[speaker_B].append(
                np.mean(similarities) if similarities else 0
            )

    # 
    
    
    responsivity_df = pd.DataFrame(
        responsivity_matrix, index=speaker_names, columns=speaker_names
    )

    return responsivity_df, time_varying_responsivity


def compute_responsivity_by_conversation(df, window_size=4, decay_rate=0.1):
    """
    Compute responsivity metrics grouped by conversation_id.

    Args:
        df (pd.DataFrame): DataFrame containing conversation data.
        window_size (int): Time window for considering previous turns.
        decay_rate (float): Exponential decay rate for weighting past turns.

    Returns:
        dict: Dictionary of responsivity results per conversation.
    """
    grouped = df.groupby("conversation_id")
    responsivity_results = {}

    for conversation_id, group in grouped:
        print(f"Processing conversation_id: {conversation_id}")
        responsivity_df, time_varying_responsivity = compute_responsivity(
            group, window_size, decay_rate
        )
        responsivity_results[conversation_id] = (
            responsivity_df,
            time_varying_responsivity,
        )

    return responsivity_results


def plot_responsivity_matrix(responsivity_df):
    """Plot the responsivity matrix as a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(responsivity_df, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Responsivity Matrix")
    plt.xlabel("Speaker B")
    plt.ylabel("Speaker A")
    plt.show()


def plot_time_varying_responsivity(time_varying_responsivity, speaker_pairs):
    """Plot time-varying responsivity signals for specified speaker pairs."""
    plt.figure(figsize=(12, 6))
    for speaker_A, speaker_B in speaker_pairs:
        plt.plot(
            time_varying_responsivity[speaker_B], label=f"{speaker_A} -> {speaker_B}"
        )
    plt.title("Time-Varying Responsivity")
    plt.xlabel("Turn Index")
    plt.ylabel("Responsivity")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\data_llama70B_processed_output.pkl"
    print("Loading data")
    df = load_data(input_path)
    df = df.dropna(subset=["Latent-Attention_Embedding"])
    print("Computing responsivity metrics by conversation")
    df["speaker_name"] = df["speaker_name"].str.lower().str.strip()
    df["speaker_name"] = df["speaker_name"].str.replace(r"\[.*\]", "", regex=True)
    df = df[
        ~df["speaker_name"].str.contains(
            "^speaker|moderator|audio|computer|computer voice|facilitator|group|highlight|interpreter|interviewer|multiple voices|other speaker|participant|redacted|speaker X|unknown|video"
        )
    ]
    df["speaker_name"] = df["speaker_name"].apply(lambda x: re.sub(r"^\s+|\s+$", "", x))

    # Dictionary to store all matrices for a consolidated output
    all_responsivity_matrices = []

    responsivity_results = compute_responsivity_by_conversation(df)

    for conversation_id, (
        responsivity_df,
        time_varying_responsivity,
    ) in responsivity_results.items():
        conversation_speakers = df[df["conversation_id"] == conversation_id][
            "speaker_name"
        ].unique()

        if len(conversation_speakers) >= 2:
            sample_pairs = [
                (conversation_speakers[i], conversation_speakers[j])
                for i in range(len(conversation_speakers))
                for j in range(i + 1, len(conversation_speakers))
            ]
        else:
            print(
                f"Not enough speakers in conversation {conversation_id} to create sample pairs."
            )
            sample_pairs = []

        print(f"Sample pairs for conversation {conversation_id}: {sample_pairs}")

        # Add conversation ID to the matrix for consolidated output
        responsivity_df["conversation_id"] = conversation_id
        responsivity_df = responsivity_df.reset_index().melt(
            id_vars=["index", "conversation_id"],
            var_name="Speaker_B",
            value_name="Responsivity",
        )
        responsivity_df.rename(columns={"index": "Speaker_A"}, inplace=True)
        all_responsivity_matrices.append(responsivity_df)

        #print(f"Plotting responsivity matrix for conversation {conversation_id}")
        #plot_responsivity_matrix(
        #    responsivity_df.pivot("Speaker_A", "Speaker_B", "Responsivity")
        #)

        """
        if sample_pairs:
            print(
                f"Plotting time-varying responsivity for conversation {conversation_id}"
            )
            plot_time_varying_responsivity(time_varying_responsivity, sample_pairs)
        else:
            print(
                f"No valid sample pairs for plotting in conversation {conversation_id}."
            )
        """

    # Consolidate all matrices into a single DataFrame
    consolidated_df = pd.concat(all_responsivity_matrices, ignore_index=True)

    # Save consolidated file
    # map the column in df based on conversation_id and speaker_name to the consolidated dataframe: is_fac
    consolidated_df["is_fac"] = consolidated_df.apply(
        lambda row: df[
            (df["conversation_id"] == row["conversation_id"]) & 
            (df["speaker_name"] == row["Speaker_A"])
        ]["is_fac"].iloc[0],
        axis=1
    )
    consolidated_df = consolidated_df[consolidated_df["Responsivity"] != 0]
    total_output_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\total_responsivity_matrix.csv"
    consolidated_df.to_pickle(r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\total_responsivity_matrix.pkl")
    consolidated_df.to_csv(total_output_path, index=False)
    print(f"Consolidated responsivity matrix saved to {total_output_path}")
    
    
