import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import seaborn as sns
import re


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

    # Separate facilitator and participant data
    # facilitator_df = df[df["is_fac"] == True]
    facilitator_df = df[(df["is_fac"] == True) | (df["cofacilitated"] == True)]
    participant_df = df[df["is_fac"] == False]

    # Group by conversation_id
    grouped_fac = facilitator_df.groupby("conversation_id")
    grouped_part = participant_df.groupby("conversation_id")

    # Process facilitator and participant data
    for conversation_id, group in grouped_fac:
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

    for conversation_id, group in grouped_part:
        group = group.sort_values("SpeakerTurn")
        speeds = calculate_speeds(group, metric)
        for i, speed in enumerate(speeds, start=1):
            results.append(
                {
                    "conversation_id": conversation_id,
                    "turn_index": group.iloc[i]["SpeakerTurn"],
                    "semantic_speed": speed,
                    "is_fac": False,
                }
            )

    speeds_df = pd.DataFrame(results)

    # Normalize turn indices
    speeds_df["turn_index"] = speeds_df["turn_index"].astype(int)
    speeds_df["turn_normalized"] = speeds_df.groupby("conversation_id")[
        "turn_index"
    ].transform(lambda x: (x / x.max()).round(2))

    # Assign each turn to a chunk
    speeds_df["chunk"] = speeds_df["turn_normalized"].apply(
        lambda x: 0 if x < 0.25 else (1 if x < 0.75 else 2)
    )

    # Calculate mean speeds for each chunk and role
    chunk_means = (
        speeds_df.groupby(["conversation_id", "chunk", "is_fac"]).mean().reset_index()
    )

    # Replace chunk numbers with ranges
    chunk_means["chunk"] = chunk_means["chunk"].map(
        {0: "0-25%", 1: "25-75%", 2: "75-100%"}
    )

    # Overall speeds
    overall_speeds = (
        speeds_df.groupby(["conversation_id", "is_fac"])["semantic_speed"]
        .mean()
        .reset_index()
    )
    overall_speeds = overall_speeds.rename(
        columns={"semantic_speed": "semantic_speed_overall"}
    )
    
    # split semantic speed overall into facilitator and participant and put as column
    overall_speeds = overall_speeds.pivot(index="conversation_id", columns="is_fac", values="semantic_speed_overall").reset_index()
    overall_speeds = overall_speeds.rename(
        columns={True: "overall_facilitator_semantic_speed", False: "overall_participant_semantic_speed"}
    )
    
    # Merge and pivot to get metrics by role
    merged_df = pd.merge(chunk_means, overall_speeds, on=["conversation_id"])
    pivot_df = merged_df.pivot(
        index="conversation_id", columns=["chunk", "is_fac"], values="semantic_speed"
    ).reset_index()

    # Calculate facilitator vs. participants average semantic speed ratio
    overall_ratios = overall_speeds.copy()
    
    overall_ratios["fac_to_part_ratio"] = overall_ratios["overall_facilitator_semantic_speed"] / overall_ratios["overall_participant_semantic_speed"]

    pivot_df.columns = [
        "_".join(map(str, col)).strip("_") if isinstance(col, tuple) else col
        for col in pivot_df.columns
    ]

    final_df = pd.merge(
        pivot_df,
        overall_ratios[["conversation_id", "fac_to_part_ratio", "overall_facilitator_semantic_speed", "overall_participant_semantic_speed"]],
        on="conversation_id",
    )

    # Rename columns for clarity
    chunk_columns = {
        f"{chunk}_True": f"fac_{chunk}" for chunk in ["0-25%", "25-75%", "75-100%"]
    }
    chunk_columns.update(
        {f"{chunk}_False": f"part_{chunk}" for chunk in ["0-25%", "25-75%", "75-100%"]}
    )

    final_df = final_df.rename(columns=chunk_columns)
    final_df = final_df.rename(columns={"overall_facilitator_semantic_speed": "facilitator_semantic_speed", "overall_participant_semantic_speed": "participant_semantic_speed"})
    return final_df


def visualize_semantic_speed(speeds_df):
    """
    Visualize semantic speed for facilitators per conversation.

    Parameters:
    - speeds_df: DataFrame containing semantic speeds.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=speeds_df,
        x="turn_index",
        y="semantic_speed",
        hue="conversation_id",
        marker="o",
    )
    plt.title("Semantic Speed Across Turns for Facilitators")
    plt.xlabel("Turn Index")
    plt.ylabel("Semantic Speed")
    plt.grid(True)
    plt.legend(title="Conversation ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load data
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\data_llama70B_processed_output.pkl"
    print("Loading data")
    df = pd.read_pickle(input_path)

    # Preprocess Data
    df = df.dropna(subset=["Latent-Attention_Embedding"])
    df["speaker_name"] = df["speaker_name"].str.lower().str.strip()
    df["speaker_name"] = df["speaker_name"].str.replace(r"\[.*\]", "", regex=True)
    df = df[
        ~df["speaker_name"].str.contains(
            "^speaker|moderator|audio|computer|computer voice|facilitator|group|highlight|interpreter|interviewer|multiple voices|other speaker|participant|redacted|speaker X|unknown|video"
        )
    ]
    df["speaker_name"] = df["speaker_name"].apply(lambda x: re.sub(r"^\s+|\s+$", "", x))

    # Calculate Semantic Speed
    print("Calculating semantic speed")
    speeds_df = calculate_semantic_speed_per_conversation(df, metric="cosine")

    # Display and Visualize
    # print(speeds_df.head())
    # visualize_semantic_speed(speeds_df)

    # Save to CSV
    output_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\semantic_speed.csv"
    speeds_df.to_csv(output_path, index=False)

    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\facilitators_features_big.csv"
    to_merge = pd.read_csv(input_path)
    # kill every column in to_merge that ends either with _y or _x
    to_merge = to_merge[
        [col for col in to_merge.columns if not col.endswith("_y") and not col.endswith("_x")]
    ]
    
    to_merge = to_merge.merge(speeds_df, on="conversation_id")
    to_merge.to_csv(input_path, index=False)
