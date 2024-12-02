import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt


def gini_coefficient(values):
    """
    Computes the Gini coefficient for a given array of non-negative values.

    Args:
        values (array-like): Array of resource shares per participant.

    Returns:
        float: Gini coefficient (0 to 1), where higher values indicate more inequality.
    """
    values = np.array(values, dtype=float)
    if len(values) <= 1 or np.any(values < 0):
        return 0.0

    values = np.sort(values)  # Sort the values
    n = len(values)
    cumulative_sum = np.cumsum(values)
    mean_value = np.mean(values)

    if mean_value == 0:
        return 0.0

    # Proper formula for Gini coefficient
    gini_numerator = np.sum((2 * np.arange(1, n + 1) - n - 1) * values)
    gini_denominator = n * np.sum(values)
    return gini_numerator / gini_denominator


def calculate_gini_coefficients(conversations):
    """
    Calculates Gini coefficients for speaking time (Gd) and speaking turns (Gc) per conversation.
    Args:
        conversations (pd.DataFrame): DataFrame with columns `conversation_id`, `speaker_name`,
                                      `duration` (speaking time), and `turn_count` (turns taken).

    Returns:
        pd.DataFrame: Gini coefficients per conversation, with columns `conversation_id`, `Gd`, `Gc`.
    """
    gini_results = []

    # Group by conversation
    for conversation_id, group in conversations:
        # Total speaking time per participant
        duration_per_speaker = group.groupby("speaker_name")["duration"].sum().values

        # Total turn count per participant
        turns_per_speaker = group.groupby("speaker_name")["SpeakerTurn"].count().values

        # Calculate Gini coefficients
        Gd = gini_coefficient(duration_per_speaker)
        Gc = gini_coefficient(turns_per_speaker)

        # Store results
        gini_results.append({"conversation_id": conversation_id, "Gd": Gd, "Gc": Gc})

    return pd.DataFrame(gini_results)


def plot_gini_coefficients(gini_df):
    """
    Plots the Gini coefficients (Gd and Gc) for all conversations.
    Args:
        gini_df (pd.DataFrame): DataFrame containing Gini coefficients with columns `Gd` and `Gc`.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(gini_df["Gd"], gini_df["Gc"], alpha=0.7, edgecolor="k")
    plt.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="G = 0.5")
    plt.axvline(0.5, color="blue", linestyle="--", linewidth=0.8)
    plt.title("Spread of Gini Coefficients (Gd and Gc)", fontsize=14)
    plt.xlabel("Gd: Gini Coefficient for Duration", fontsize=12)
    plt.ylabel("Gc: Gini Coefficient for Turn Count", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


def load_data(input_path):
    """Load data from a pickle file."""
    print(f"Loading data from {input_path}")
    return pd.read_pickle(input_path)


def prepare_data(df):
    df["speaker_name"] = df["speaker_name"].str.lower().str.strip()
    df["speaker_name"] = df["speaker_name"].str.replace(r"\[.*\]", "", regex=True)
    df = df[
        ~df["speaker_name"].str.contains(
            "^speaker|moderator|audio|computer|computer voice|facilitator|group|highlight|interpreter|interviewer|multiple voices|other speaker|participant|redacted|speaker X|unknown|video"
        )
    ]
    df["speaker_name"] = df["speaker_name"].apply(lambda x: re.sub(r"^\s+|\s+$", "", x))
    return df


def calculate_semantic_speed_per_conversation(df, metric="cosine"):
    """
    Calculate semantic speed for facilitators and participants within each conversation_id.

    Parameters:
    - df: DataFrame containing conversation data.
    - metric: Distance metric ('cosine' or 'euclidean').

    Returns:
    - speeds_df: DataFrame with semantic speeds per turn and conversation.
    """

    def calculate_speeds(group, metric):
        """Calculate semantic speeds for a given group."""
        embeddings = np.stack(group["Latent-Attention_Embedding"])
        speeds = []
        for t in range(1, len(embeddings)):
            if metric == "cosine":
                dist = cosine_distances([embeddings[t - 1]], [embeddings[t]])[0][0]
            elif metric == "euclidean":
                dist = np.linalg.norm(embeddings[t] - embeddings[t - 1])
            else:
                raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")
            speeds.append(dist)
        return speeds

    results = []

    # Process facilitator and participant data
    for conversation_id, group in df:
        group = group.sort_values("SpeakerTurn")
        speeds = calculate_speeds(group, metric)
        for i, speed in enumerate(speeds, start=1):
            results.append(
                {
                    "conversation_id": conversation_id,
                    "turn_index": group.iloc[i]["SpeakerTurn"],
                    "semantic_speed": speed,
                    "is_fac": True,
                }
            )

    speeds_df = pd.DataFrame(results)

    # Normalize turn indices
    speeds_df["turn_index"] = speeds_df["turn_index"].astype(int)
    speeds_df["turn_normalized"] = speeds_df.groupby("conversation_id")[
        "turn_index"
    ].transform(lambda x: (x / x.max()).round(2))
    return speeds_df


def plot_speeds(df):
    num_points = 100
    normalized_turns = np.linspace(0, 1, num_points)
    
    # Initialize an array to hold the summed semantic speeds
    summed_speeds = np.zeros(num_points)
    
    # Loop over each conversation
    for conversation_id, group in df.groupby("conversation_id"):
        # Get the data for this conversation
        x = group["turn_normalized"].values
        y = group["semantic_speed"].values
        
        # Interpolate onto the common normalized_turns
        y_interp = np.interp(normalized_turns, x, y)
        
        # Add to the summed_speeds
        summed_speeds += y_interp
        
    # Plot the summed semantic speeds
    plt.figure(figsize=(12, 8))
    plt.plot(normalized_turns, summed_speeds, label='Overall Speed')
    plt.xlabel("Normalized Turn Index")
    plt.ylabel("Summed Semantic Speed")
    plt.title("Overall Semantic Speed")
    plt.legend()
    plt.grid(True)
    plt.show()


def conversational_structure(df):
    # Step 1: Calculate conversation-level metrics
    grouped = df.groupby("conversation_id")
    metrics = pd.DataFrame({
        "num_speakers": grouped["speaker_name"].nunique(),
        "num_turns": grouped.size(),
        "conversation_length": grouped["audio_end_offset"].max() - grouped["audio_start_offset"].min(),
        "personal_sharing": grouped["Personal experience"].sum() + grouped["Personal story"].sum(),
        "personal_experience": grouped["Personal experience"].sum(),
        "personal_story": grouped["Personal story"].sum(),
        "average_semantic_speed": calculate_semantic_speed_per_conversation(grouped).groupby("conversation_id")["semantic_speed"].mean(),
    }).reset_index()

    iq = calculate_gini_coefficients(grouped)
    metrics = metrics.merge(iq, on="conversation_id", how="left")

    speaker_info = df[["conversation_id", "collection_id"]].drop_duplicates()
    
    # create a column is_fac and cofacilitated and enter the name(s) of the facilitator(s)
    facilitators = df[df["is_fac"] == True].groupby("conversation_id")["speaker_name"].unique().reset_index()
    facilitators.columns = ["conversation_id", "facilitators"]
    facilitators["is_fac"] = facilitators["facilitators"].apply(lambda x: len(x) > 0)
    facilitators["facilitators"] = facilitators["facilitators"].apply(lambda x: ", ".join(x))

    
    cofacilitators = df[df["cofacilitated"] == 1.0].groupby("conversation_id")["speaker_name"].unique().reset_index()
    cofacilitators.columns = ["conversation_id", "cofacilitators"]
    cofacilitators["is_fac"] = cofacilitators["cofacilitators"].apply(lambda x: len(x) > 0)
    cofacilitators["cofacilitators"] = cofacilitators["cofacilitators"].apply(lambda x: ", ".join(x))
    facilitators = facilitators.merge(cofacilitators, on="conversation_id", how="left")
    
    speaker_info = speaker_info.merge(facilitators, on="conversation_id", how="left")
    
    final = speaker_info.merge(metrics, on="conversation_id", how="left")

    # drop every facilitators row where nan
    final = final.dropna(subset=["facilitators"])
    final = final.drop(columns=["is_fac_y", "is_fac_x"])
    
    return final


if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\data_llama70B_processed_output.pkl"
    print("Loading data")
    df = load_data(input_path)
    # drop na in Latent-Attention_Embedding
    df = df.dropna(subset=["Latent-Attention_Embedding"])
    df = prepare_data(df)
    cs = conversational_structure(df)
    # save cs to .csv
    cs.to_csv(r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure.csv")