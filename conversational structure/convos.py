import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import seaborn as sns


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
    mean_value = np.mean(values)

    if mean_value == 0:
        return 0.0

    # Proper formula for Gini coefficient
    gini_numerator = np.sum((2 * np.arange(1, n + 1) - n - 1) * values)
    gini_denominator = n * np.sum(values)
    return gini_numerator / gini_denominator


def calculate_gini_coefficients(conversations):
    """
    Calculates Gini coefficients for speaking time (Gd) and turn count (Gc) per conversation, 
    excluding all facilitator turns (where is_fac is True).

    Args:
        conversations (pd.core.groupby.generic.DataFrameGroupBy): GroupBy object with columns 
            `conversation_id`, `speaker_name`, `duration`, and `SpeakerTurn`.

    Returns:
        pd.DataFrame: DataFrame with columns `conversation_id`, `Gd`, and `Gc`.
    """
    gini_results = []

    # Group by conversation
    for conversation_id, group in conversations:
        # Exclude facilitator turns
        group = group[~group["is_fac"]]
        
        # Skip conversation if no participant turns remain
        if group.empty:
            continue

        # Total speaking time per participant
        duration_per_speaker = group.groupby("speaker_name")["duration"].sum().values

        # Total turn count per participant
        turns_per_speaker = group.groupby("speaker_name")["SpeakerTurn"].count().values

        # Calculate Gini coefficients
        Gd = gini_coefficient(duration_per_speaker)
        Gc = gini_coefficient(turns_per_speaker)

        # Store results with Gd and Gc
        gini_results.append({
            "conversation_id": conversation_id, 
            "Gd": Gd, 
            "Gc": Gc
        })

    return pd.DataFrame(gini_results)


def plot_gini_coefficients(gini_df):
    """
    Plots the Gini coefficients (Gd for duration and Gc for turn count)
    for all conversations.
    
    Args:
        gini_df (pd.DataFrame): DataFrame containing Gini coefficients with columns `Gd` and `Gc`.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(gini_df["Gd"], gini_df["Gc"], alpha=0.7, edgecolor="k")
    plt.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Equity = 0.5")
    plt.axvline(0.5, color="blue", linestyle="--", linewidth=0.8)
    plt.title("Spread of Gini Coefficients (Duration and Turn Taking)", fontsize=14)
    plt.xlabel("Gd: Gini Coefficient for Duration", fontsize=12)
    plt.ylabel("Gc: Gini Coefficient for Turn Count", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()  # Adjust layout
    plt.show()


def load_data(input_path):
    """Load data from a CSV file."""
    print(f"Loading data from {input_path}")
    return pd.read_csv(input_path)


def prepare_data(df):
    # Calculate SpeakerTurn for each speaker within a conversation
    df["SpeakerTurn"] = df.groupby(["conversation_id", "speaker_name"]).cumcount() + 1

    # Identify max SpeakerTurn for is_fac speakers within each conversation
    is_fac_turns = (
        df[df["is_fac"] == True]
        .groupby(["conversation_id", "speaker_name"])["SpeakerTurn"]
        .max()
        .reset_index()
    )

    # Count the number of unique facilitators (is_fac) in each conversation
    fac_counts = (
        df[df["is_fac"] == True]
        .groupby("conversation_id")["speaker_name"]
        .nunique()
        .reset_index(name="fac_count")
    )

    # Merge facilitator counts back into the main dataframe
    df = pd.merge(df, fac_counts, on="conversation_id", how="left")

    # Identify conversations with multiple facilitators
    multi_fac_conversations = fac_counts[fac_counts["fac_count"] > 1]["conversation_id"]

    # Filter facilitator SpeakerTurn data for multi-facilitator conversations
    multi_fac_turns = is_fac_turns[is_fac_turns["conversation_id"].isin(multi_fac_conversations)]

    # Identify conversations where all facilitators have SpeakerTurn below the threshold
    threshold = 3
    short_multi_fac_conversations = (
        multi_fac_turns.groupby("conversation_id")["SpeakerTurn"]
        .max()
        .reset_index()
    )
    short_multi_fac_conversations = short_multi_fac_conversations[
        short_multi_fac_conversations["SpeakerTurn"] < threshold
    ]["conversation_id"]

    # Handle single-facilitator conversations
    single_fac_conversations = fac_counts[fac_counts["fac_count"] == 1]["conversation_id"]
    single_fac_turns = is_fac_turns[is_fac_turns["conversation_id"].isin(single_fac_conversations)]
    short_single_fac_conversations = single_fac_turns[
        single_fac_turns["SpeakerTurn"] < threshold
    ]["conversation_id"]

    # Redact conversations: drop if single facilitator is below the threshold,
    # or if all facilitators in multi-facilitator conversations are below the threshold
    redact_conversations = set(short_single_fac_conversations).union(
        set(short_multi_fac_conversations)
    )
    df = df[~df["conversation_id"].isin(redact_conversations)]

    # For multi-facilitator conversations: if one facilitator is above the threshold,
    # set is_fac = False for those below the threshold
    if not multi_fac_turns.empty:
        to_update = multi_fac_turns[
            multi_fac_turns["SpeakerTurn"] < threshold
        ]
        to_update = to_update[~to_update["conversation_id"].isin(short_multi_fac_conversations)]

        for _, row in to_update.iterrows():
            df.loc[
                (df["conversation_id"] == row["conversation_id"]) & 
                (df["speaker_name"] == row["speaker_name"]),
                "is_fac"
            ] = False
    return df


def conversational_structure(df):
    # Step 1: Calculate conversation-level metrics
    # Grouping on the full dataframe for most metrics
    grouped = df.groupby("conversation_id")
    
    # Create a filtered dataframe excluding facilitator turns for counting turns
    non_fac_df = df[~df["is_fac"]]
    non_fac_grouped = non_fac_df.groupby("conversation_id")
    
    metrics = pd.DataFrame({
        "num_speakers": grouped["speaker_name"].nunique(),
        # Now num_turns counts only non-facilitator turns
        "num_turns": non_fac_grouped.size(),
        "conversation_length": grouped["audio_end_offset"].max() - grouped["audio_start_offset"].min(),
        "personal_sharing": non_fac_grouped["Personal experience"].sum() + non_fac_grouped["Personal story"].sum(),
        "personal_experience": non_fac_grouped["Personal experience"].sum(),
        "personal_story": non_fac_grouped["Personal story"].sum(),
    }).reset_index()
    
    speaker_info = df[["conversation_id", "collection_id"]].drop_duplicates()
    
    # Create a column for facilitators and cofacilitators
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
    
    # Drop rows where facilitator info is missing
    final = final.dropna(subset=["facilitators"])
    final = final.drop(columns=["is_fac_y", "is_fac_x"])
    
    # Plot correlation matrix (numeric columns only)
    numeric_cols = final.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()
    
    return final


if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\LLM as Judge\final.csv"
    print("Loading data")
    df = load_data(input_path)
    
    # Prepare the data
    df = prepare_data(df)
    
    # Compute conversational structure
    cs = conversational_structure(df)
    
    # Calculate Gini coefficients and rename the turn coefficient to Gc
    grouped = df.groupby("conversation_id")
    gini_df = calculate_gini_coefficients(grouped)
    gini_df = gini_df.rename(columns={"Equitable_turn_distribution": "Gc"})
    
    # Merge Gini coefficients into the conversational structure dataframe
    cs = cs.merge(gini_df, on="conversation_id", how="left")
    
    # Save the merged dataframe with Gini coefficients (Gd and Gc) included
    cs.to_csv(r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure.csv", index=False)
    
    # Optionally, display the scatter plot for Gini coefficients
    plot_gini_coefficients(gini_df)

