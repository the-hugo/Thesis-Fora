import umap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import re
from scipy.stats.mstats import winsorize
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt


def load_data(input_path):
    """Load data from a pickle file."""
    print(f"Loading data from {input_path}")
    return pd.read_pickle(input_path)


def load_csv(input_path):
    """Load data from a pickle file."""
    print(f"Loading data from {input_path}")
    return pd.read_csv(input_path)


def prepare_dfs(df, df_w_words):
    for dataframe in [df, df_w_words]:
        dataframe["speaker_name"] = dataframe["speaker_name"].str.lower().str.strip()
        dataframe["speaker_name"] = dataframe["speaker_name"].str.replace(r"\[.*\]", "", regex=True)
        dataframe = dataframe[
            ~dataframe["speaker_name"].str.contains(
                "^speaker|moderator|audio|computer|computer voice|facilitator|group|highlight|interpreter|interviewer|multiple voices|other speaker|participant|redacted|speaker X|unknown|video"
            )
        ]
        dataframe["speaker_name"] = dataframe["speaker_name"].apply(lambda x: re.sub(r"^\s+|\s+$", "", x))
    return df, df_w_words

if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\facilitators_features_clustered.csv"
    print("Loading data")
    df = load_csv(input_path)
    df_w_words = load_data(
        r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\data_llama70B_processed_output.pkl"
    )

    df, df_w_words = prepare_dfs(df, df_w_words)

    # merge the two dataframes on conversation_id and speaker name
    df = df.merge(df_w_words, on=["conversation_id", "speaker_name"], suffixes=('', '_drop'))
    df = df[[col for col in df.columns if not col.endswith('_drop')]]

    # Save the clustered data
    output_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\merged_facilitators_features_clustered.csv"
    df.to_csv(output_path, index=False)

    #split the data into clusters
    clusters = df["cluster"].unique()
    for cluster in clusters:
        df_cluster = df[df["cluster"] == cluster]
        output_path = f"C:\\Users\\paul-\\Documents\\Uni\\Management and Digital Technologies\\Thesis Fora\\Code\\data\\output\\annotated\\facilitators_features_clustered_{cluster}.csv"
        df_cluster.to_csv(output_path, index=False)