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


if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\facilitators_features_big.csv"
    print("Loading data")
    responsivity = load_data(
        r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\total_responsivity_matrix.pkl"
    )
    all_features = load_data(input_path)

    print(responsivity.columns)

    filtered_data = responsivity[responsivity["Responsivity"] != 0]

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
    # all_features = all_features.drop(columns=["Speaker_A", "Speaker_B", "is_fac"])
    print(len(all_features))
    all_features = all_features.dropna()
    print(len(all_features))
    print(all_features.columns)
    all_features.to_csv(
        r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\facilitators_features_big.csv"
    )
