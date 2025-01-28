import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_data(input_path):
    """Load data from a pickle file."""
    print(f"Loading data from {input_path}")
    return pd.read_csv(input_path)


def calculate_correlation(df):

    correlation_matrix = df.corr()

    correlation_threshold = 0.15
    high_correlation_vars = correlation_matrix.columns[
        (abs(correlation_matrix) > correlation_threshold).any()
    ]
    df = df[high_correlation_vars]

    filtered_correlation_matrix = correlation_matrix[
        (abs(correlation_matrix) > correlation_threshold).any(axis=1)
    ][high_correlation_vars]

    print(filtered_correlation_matrix)

    plt.figure(figsize=(10, 10))
    sns.heatmap(filtered_correlation_matrix, annot=True)
    plt.show()


def compute_ratios(group):
    # Facilitator data
    fac_data = group[group["is_fac"]]
    Fd = fac_data["duration"].mean()
    Fc = fac_data["SpeakerTurn"].mean()

    # Participant data
    part_data = group[~group["is_fac"]]
    Pd_avg = part_data["duration"].mean()
    Pc_avg = part_data["SpeakerTurn"].mean()

    # Calculate ratios
    Rd = Fd / Pd_avg if Pd_avg > 0 else None
    Rc = Fc / Pc_avg if Pc_avg > 0 else None

    # Assign ratios to all rows in the group
    group["Rd"] = Rd
    group["Rc"] = Rc
    return group


def aggregate_conv(df, participants):
    # calculate the rate of Facilitator to Participant change. How often do the roles change in turn taking over all turn?
    df = df.sort_values(by=["conversation_id", "SpeakerTurn"])

    # Identify role changes (True if role changes between consecutive turns)
    df["role_change"] = (
        df.groupby("conversation_id")["is_fac"]
        .apply(lambda x: x != x.shift(1))
        .reset_index(level=0, drop=True)
    )

    # Count role changes per conversation
    role_change_counts = (
        df.groupby("conversation_id")["role_change"].sum().reset_index()
    )
    role_change_counts = role_change_counts.rename(
        columns={"role_change": "role_change_count"}
    )

    # Total turns per conversation
    total_turns = df.groupby("conversation_id")["SpeakerTurn"].count().reset_index()
    total_turns = total_turns.rename(columns={"SpeakerTurn": "total_turns"})

    # Merge counts and calculate the role change rate
    role_change_rate_df = pd.merge(
        role_change_counts, total_turns, on="conversation_id"
    )
    role_change_rate_df["role_change_rate"] = role_change_rate_df[
        "role_change_count"
    ] / (role_change_rate_df["total_turns"] - 1)

    # merge the role_change_rate column to the main dataframe
    df = pd.merge(
        df,
        role_change_rate_df[["conversation_id", "role_change_rate"]],
        on="conversation_id",
    )

    # kick out all non-facilitator rows
    n_participants = not participants
    df = df[df["is_fac"] == n_participants]

    # df = df[(df["is_fac"] == n_participants) | (df["cofacilitated"] == True)]
    # kick out all non-annotated ones
    # df = df[df["annotated"]]
    # bucket fac:
    # Validation Strategies: Express appreciation, Express affirmation
    # Invitations to Participate: Open invitation, Specific invitation
    # Facilitation strategies: Provide examples, Follow up question, Make connections
    # mean: phaticity ratio	Segment	WC	Analytic	Clout	Authentic	Tone	WPS	BigWords
    # Create bucket variables

    df["Validation Strategies"] = df["Express appreciation"] + df["Express affirmation"]
    df["Invitations to Participate"] = df["Open invitation"] + df["Specific invitation"]
    df["Facilitation Strategies"] = (
        df["Provide example"] + df["Follow up question"] + df["Make connections"]
    )

    drop = [
        "Unnamed: 0",
        "Unnamed: 0.1",
        "id",
        "collection_title",
        "collection_id",
        "SpeakerTurn",
        "audio_start_offset",
        "audio_end_offset",
        "speaker_id",
        "is_fac",
        # "cofacilitated",
        "annotated",
        "start_time",
        "location",
        "source_type",
        "phatic speech",
        "words",  # Add preserved columns here since they will also be dropped
        "Latent-Attention_Embedding",
        "Segments" "Express affirmation",
        "Specific invitation",
        "Provide example",
        "Open invitation",
        "Make connections",
        "Express appreciation",
        "Follow up question",
        "Express affirmation",
    ]

    # Drop unnecessary columns, including preserved ones
    df = df.drop(columns=drop, errors="ignore")

    # Specify custom aggregation rules
    custom_agg = {
        "Validation Strategies": "sum",
        "Invitations to Participate": "sum",
        "Facilitation Strategies": "sum",
        "phaticity ratio": "mean",
        "Rd": "first",
        "Rc": "first",
        "role_change_rate": "first",
        "Personal story": "sum",
        "Personal experience": "sum",
    }

    # Add default aggregation (mean) for all other numeric columns except grouping keys
    grouping_columns = ["conversation_id", "speaker_name"]
    remaining_cols = [
        col
        for col in df.columns
        if col not in list(custom_agg.keys()) + grouping_columns
    ]
    for col in remaining_cols:
        if pd.api.types.is_numeric_dtype(
            df[col]
        ):  # Only include numeric columns for aggregation
            custom_agg[col] = "mean"

    # Perform groupby and aggregate
    df = df.groupby(grouping_columns).agg(custom_agg).reset_index()
    # keep only the row that is unique per conversation id and speaker_name
    # df = df.drop_duplicates(subset=["conversation_id", "speaker_name"])

    return df


if __name__ == "__main__":
    df = load_data(
        r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\LIWC-22 Results - data_llama70B_processed_output___ - LIWC Analysis_big.csv"
    )
    df = df.groupby("conversation_id").apply(compute_ratios).reset_index(drop=True)

    participants = False
    df = aggregate_conv(df, participants)

    # calculate_correlation(df)

    if participants:
        df.to_csv(
            r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\participants_features_big.csv"
        )
    else:
        df.to_csv(
            r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\facilitators_features_big.csv"
        )

"""
df.drop(
    columns=[
        "speaker_name",
        "speaker_id",
        "phaticity_ratio_1_count",
        "phaticity_ratio_all_count",
        "conversation_phaticity_ratio",
        "conversation_phaticity_ratio_participants",
        "conversation_id"        
    ],
    inplace=True,
)

print(df.columns)
df = df.dropna(subset=["Latent-Attention_Embedding"])

scaled_X = StandardScaler().fit_transform(
    np.vstack(df["Latent-Attention_Embedding"].values)
)
reducer = umap.UMAP(n_components=2)
embedding_2d = reducer.fit_transform(scaled_X)
# drop the Latent-Attention_Embedding column
df.drop(columns=["Latent-Attention_Embedding"], inplace=True)

# put the 2D embedding into the dataframe
df["UMAP_1"] = embedding_2d[:, 0]
df["UMAP_2"] = embedding_2d[:, 1]

df.drop(
        columns=[
            "Unnamed: 0",
            "Unnamed: 0.1",
            "id",
            "source_type",
            "collection_title",
            "annotated",
            "location",
            "words",
            "speaker_name",
            "start_time",
            "phatic speech",
            "Latent-Attention_Embedding",
            "audio_start_offset",
            "audio_end_offset",
            "Segment"
        ],
        inplace=True,
    )
"""
