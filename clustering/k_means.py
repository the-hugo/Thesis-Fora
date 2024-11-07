import umap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import re

def load_data(input_path):
    """Load data from a pickle file."""
    print(f"Loading data from {input_path}")
    return pd.read_pickle(input_path)


def aggregate_features(df):
    df["speaker_name"] = df["speaker_name"].str.lower().str.strip()
    # Removing text within brackets and unnecessary speaker labels
    df["speaker_name"] = df["speaker_name"].str.replace(r"\[.*\]", "", regex=True)
    df = df[~df["speaker_name"].str.contains(
        "^speaker|moderator|audio|computer|computer voice|facilitator|group|highlight|interpreter|interviewer|multiple voices|other speaker|participant|redacted|speaker X|unknown|video"
    )]
    df = df.drop(columns=["Unnamed: 0", "id"])

    # Handle numeric columns and clean up any extra spaces
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df["speaker_name"] = df["speaker_name"].apply(lambda x: re.sub(r'^\s+|\s+$', '', x))

    # Handle Latent_Attention_Embedding by converting lists to arrays if not already
    df["Latent_Attention_Embedding"] = df["Latent_Attention_Embedding"].apply(
        lambda x: np.array(x) if isinstance(x, list) else x
    )

    # Group by speaker name, aggregating numeric columns and Latent_Attention_Embedding
    aggregated_df = df.groupby("speaker_name").agg({
        **{col: 'mean' for col in numeric_cols},
        "Latent_Attention_Embedding": lambda x: np.mean(np.vstack(x), axis=0)
    }).reset_index()

    return aggregated_df


def apply_umap(df):
    # Select all features except 'speaker_name'
    features = df.drop(columns=["speaker_name"]).values
    latent_attention_embeddings = np.vstack(df['Latent_Attention_Embedding'].values)
    other_features = df.drop(columns=["speaker_name", "Latent_Attention_Embedding"]).values
    combined_features = np.hstack((latent_attention_embeddings, other_features))
    scaled_X = StandardScaler().fit_transform(combined_features)

    # Set the number of UMAP components
    n_components = 2
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(scaled_X)

    # Add UMAP dimensions to df
    for i in range(n_components):
        df[f"umap_{i}"] = embedding[:, i]

    return df


def plot_clusters(df):
    fig = px.scatter(
        df, x="umap_0", y="umap_1", color="cluster", hover_name="speaker_name"
    )
    fig.show()


def k_means(df):
    # Use UMAP features only for clustering
    umap_features = [col for col in df.columns if col.startswith("umap_")]
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[umap_features])
    return df


if __name__ == "__main__":
    input_path = "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/output/annotated/output_filled_phatic_ratio.pkl"
    print("Loading data")
    df = load_data(input_path)
    df = df.dropna(subset=["Latent_Attention_Embedding"])
    df = df[df["is_fac"] == False]

    print("Aggregating features by speaker")
    df = aggregate_features(df)

    print("Applying UMAP to all features")
    df = apply_umap(df)

    print("K-means clustering")
    df = k_means(df)

    print("Plotting clusters by speaker")
    plot_clusters(df)
