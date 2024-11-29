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
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    else:
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

    responsivity_df = pd.DataFrame(
        responsivity_matrix, index=speaker_names, columns=speaker_names
    )

    return responsivity_df, time_varying_responsivity


def compute_responsivity_by_conversation(df, window_size=6, decay_rate=0.1):
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
    # plt.show()


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
    # plt.show()


if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\facilitators_features_big.csv"
    print("Loading data")
    responsivity = load_data(
        r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\total_responsivity_matrix.pkl"
    )
    all_features = load_data(input_path)

    # calculate the overall responsivity by taking the mean but only if speaker a and speaker b are different
    print(responsivity.columns)

    filtered_data = responsivity[
        (responsivity["Speaker_A"] != responsivity["Speaker_B"])
        & (responsivity["is_fac"] == True)
    ]

    overall_responsivity = filtered_data.groupby("conversation_id")[
        "Responsivity"
    ].mean()
    # now merke the overall responsivity to the features by conversation_id and speaker_name
    all_features["conversation_id"] = all_features["conversation_id"].astype(int)
    overall_responsivity = overall_responsivity.reset_index()
    overall_responsivity["conversation_id"] = overall_responsivity[
        "conversation_id"
    ].astype(int)
    # kill all columns in all features that start with responivity
    all_features = all_features[
        [col for col in all_features.columns if not col.startswith("Responsivity")]
    ]
    # do the same for unnamed
    all_features = all_features[
        [col for col in all_features.columns if not col.startswith("Unnamed")]
    ]
    all_features = all_features.merge(overall_responsivity, on="conversation_id")
    #all_features = all_features.drop(columns=["Speaker_A", "Speaker_B", "is_fac"])
    print(len(all_features))
    all_features = all_features.dropna()
    print(len(all_features))
    print(all_features.columns)
    all_features.to_csv(
        r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\facilitators_features_big.csv"
    )
