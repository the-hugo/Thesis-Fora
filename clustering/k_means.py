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
    return pd.read_csv(input_path)


def aggregate_features(df):
    df["speaker_name"] = df["speaker_name"].str.lower().str.strip()
    # Removing text within brackets and unnecessary speaker labels
    df["speaker_name"] = df["speaker_name"].str.replace(r"\[.*\]", "", regex=True)
    df = df[
        ~df["speaker_name"].str.contains(
            "^speaker|moderator|audio|computer|computer voice|facilitator|group|highlight|interpreter|interviewer|multiple voices|other speaker|participant|redacted|speaker X|unknown|video"
        )
    ]
    df = df.drop(columns=["Unnamed: 0", "id"])

    # Handle numeric columns and clean up any extra spaces
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df["speaker_name"] = df["speaker_name"].apply(lambda x: re.sub(r"^\s+|\s+$", "", x))

    # Handle Latent_Attention_Embedding by converting lists to arrays if not already
    df["Latent_Attention_Embedding"] = df["Latent_Attention_Embedding"].apply(
        lambda x: np.array(x) if isinstance(x, list) else x
    )

    # Group by speaker name, aggregating numeric columns and Latent_Attention_Embedding
    aggregated_df = (
        df.groupby("speaker_name")
        .agg(
            {
                **{col: "mean" for col in numeric_cols},
                "Latent_Attention_Embedding": lambda x: np.mean(np.vstack(x), axis=0),
            }
        )
        .reset_index()
    )

    return aggregated_df


def preprocessing(df):
    """
    features = df.drop(columns=["speaker_name"]).values
    latent_attention_embeddings = np.vstack(df['Latent_Attention_Embedding'].values)
    other_features = df.drop(columns=["speaker_name", "Latent_Attention_Embedding"]).values
    combined_features = np.hstack((latent_attention_embeddings, other_features))
    """

    combined_features = df.drop(columns=["speaker_name", "conversation_id"])

    rows_before = combined_features.shape[0]
    combined_features = combined_features.dropna()
    rows_after = combined_features.shape[0]
    rows_dropped = rows_before - rows_after
    print(f"Number of rows dropped: {rows_dropped}")

    combined_features = prepare_data(combined_features)
    return combined_features


def apply_umap(combined_features):

    scaled_X = StandardScaler().fit_transform(combined_features)

    # Set the number of UMAP components
    n_components = 2
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(scaled_X)

    # Add UMAP dimensions to df
    for i in range(n_components):
        combined_features[f"umap_{i}"] = embedding[:, i]
    return combined_features


def plot_clusters(df):
    fig = px.scatter(df, x="umap_0", y="umap_1", color="cluster", hover_name="cluster")
    fig.show()


def k_means(df, n_clusters):
    # Use UMAP features only for clustering
    umap_features = df.columns
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[umap_features])
    centroids = kmeans.cluster_centers_
    show_centroids(centroids, df, n_clusters)
    return df


def prepare_data(df):
    # Drop the 'WC' column if it exists
    if "WC" in df.columns:
        df = df.drop(columns=["WC"])
    # Winsorize the data to handle outliers
    # Normalize the data to have zero mean and unit variance
    df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    # Apply winsorization to each column
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = winsorize(df[col], limits=[0.05, 0.05])

    # Assess multicollinearity using Variance Inflation Factor (VIF)

    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    print("VIF Data:\n", vif_data)

    # Assess collinearity using Pearson correlations
    correlation_matrix = df.corr()
    # print("Correlation Matrix:\n", correlation_matrix)

    return df


def determine_cluster(df, max_clusters=10, random_state=42):
    """
    Determine the optimal number of clusters for a dataset using the Elbow Method and Silhouette Score.

    Parameters:
        df (pd.DataFrame): The input data (only numerical columns should be included).
        max_clusters (int): The maximum number of clusters to evaluate.
        random_state (int): The random state for reproducibility.

    Returns:
        int: The optimal number of clusters.
    """
    wcss = []  # Within-cluster sum of squares
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df, kmeans.labels_))

    # Plot the Elbow Method
    plt.figure(figsize=(10, 5))
    plt.plot(cluster_range, wcss, marker="o", linestyle="--")
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS (Within Cluster Sum of Squares)")
    # plt.show()

    # Plot the Silhouette Score
    plt.figure(figsize=(10, 5))
    plt.plot(cluster_range, silhouette_scores, marker="o", linestyle="--")
    plt.title("Silhouette Score for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    # plt.show()

    # Determine the optimal number of clusters based on the maximum Silhouette Score
    optimal_clusters = cluster_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"Optimal number of clusters based on Silhouette Score: {optimal_clusters}")

    return optimal_clusters


def show_centroids(centroids, df, n_clusters):
    feature_columns = df.columns[df.columns != "cluster"]
    centroids_df = pd.DataFrame(centroids, columns=feature_columns)
    centroids_df.index.name = "Cluster"

    feature_means = df.mean()

    for cluster_idx in range(n_clusters):
        plt.figure(figsize=(12, 6))
        deviations = centroids_df.iloc[cluster_idx] - feature_means
        deviations.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.axhline(0, color="gray", linestyle="--")
        plt.title(f"Cluster {cluster_idx} Feature Deviations from Mean")
        plt.xlabel("Features")
        plt.ylabel("Deviation from Mean")
        plt.tight_layout()
        plt.show()
    return centroids_df


if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\participants_features.csv"
    print("Loading data")
    df = load_data(input_path)
    # df = df.dropna(subset=["Latent_Attention_Embedding"])
    # df = df[df["is_fac"] == False]

    # print("Aggregating features by speaker")
    # df = aggregate_features(df)

    # Exclude specific columns
    exclude_columns = ["Facilitation_Strategies", "Invitations_to_Participate", "Validation_Strategies"]
    #exclude_columns = ["Rc", "Rd"]
    df = df.drop(columns=exclude_columns)

    print("Applying UMAP to all features")
    df = preprocessing(df)

    n_clusters = determine_cluster(df)

    print("K-means clustering")
    df = k_means(df, n_clusters)

    df = apply_umap(df)
    plot_clusters(df)
    
    # Save the clustered data
    output_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\participants_features_clustered.csv"
    df.to_csv(output_path, index=False)