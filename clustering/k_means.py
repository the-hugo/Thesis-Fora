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
from sklearn.decomposition import PCA  # <-- Added for PCA

def load_data(input_path):
    """Load data from a pickle file."""
    print(f"Loading data from {input_path}")
    return pd.read_csv(input_path)


def aggregate_features(df):
    df["speaker_name"] = df["speaker_name"].str.lower().str.strip()
    df["speaker_name"] = df["speaker_name"].str.replace(r"\[.*\]", "", regex=True)
    df = df[
        ~df["speaker_name"].str.contains(
            "^speaker|moderator|audio|computer|computer voice|facilitator|group|highlight|interpreter|interviewer|multiple voices|other speaker|participant|redacted|speaker X|unknown|video"
        )
    ]
    df["speaker_name"] = df["speaker_name"].apply(lambda x: re.sub(r"^\s+|\s+$", "", x))

    df = df.drop(columns=["Unnamed: 0", "id"])

    # Handle numeric columns and clean up any extra spaces
    numeric_cols = df.select_dtypes(include=[np.number]).columns

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
    # Drop columns ending with _x or _y if their counterparts exist
    for col in df.columns:
        if col.endswith('_x') and col[:-2] + '_y' in df.columns:
            df = df.drop(columns=[col])
            df = df.rename(columns={col[:-2] + '_y': col[:-2]})
        elif col.endswith('_y') and col[:-2] + '_x' in df.columns:
            df = df.drop(columns=[col])
            df = df.rename(columns={col[:-2] + '_x': col[:-2]})
    
    # combined_features = df.drop(columns=["speaker_name", "conversation_id"])
    combined_features = df.copy()
    rows_before = combined_features.shape[0]
    combined_features = combined_features.dropna()
    rows_after = combined_features.shape[0]
    rows_dropped = rows_before - rows_after
    print(f"Number of rows dropped: {rows_dropped}")

    combined_features = prepare_data(combined_features)
    return combined_features


def apply_umap(combined_features):
    # Preserve 'conversation_id' and 'speaker_name' columns
    preserved_columns = combined_features[
        ["conversation_id", "speaker_name"]
    ].reset_index(drop=True)
    combined_features = combined_features.drop(
        columns=["conversation_id", "speaker_name"]
    ).reset_index(drop=True)

    scaled_X = StandardScaler().fit_transform(combined_features)

    # Set the number of UMAP components
    n_components = 2
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(scaled_X)

    # Add UMAP dimensions to combined_features
    for i in range(n_components):
        combined_features[f"umap_{i}"] = embedding[:, i]

    # Add back the preserved columns
    combined_features = pd.concat([preserved_columns, combined_features], axis=1)
    return combined_features


def plot_clusters(df):
    fig = px.scatter(
        df,
        x="umap_0",
        y="umap_1",
        color="cluster",
        hover_name="speaker_name",
    )
    # add colors
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(
        title="UMAP Clustering",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        legend_title="Cluster",
        # use a color mode without yellow
        coloraxis=dict(
            colorscale=[
                [0.0, "rgb(0,0,255)"],
                [0.5, "rgb(0,255,0)"],
                [1.0, "rgb(255,0,0)"],
            ]
        ),
    )

    fig.show()


def k_means(df, n_clusters):
    # Preserve 'conversation_id' and 'speaker_name' columns
    preserved_columns = df[["conversation_id", "speaker_name"]].reset_index(drop=True)
    df = df.drop(columns=["conversation_id", "speaker_name"]).reset_index(drop=True)

    # Use all numeric features for clustering (UMAP features will be added later)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(df)
    centroids = kmeans.cluster_centers_
    show_centroids(centroids, df, n_clusters)

    # Add back the preserved columns
    df = pd.concat([preserved_columns, df], axis=1)
    return df


def prepare_data(df):
    df["speaker_name"] = df["speaker_name"].str.lower().str.strip()
    df["speaker_name"] = df["speaker_name"].str.replace(r"\[.*\]", "", regex=True)
    df = df[
        ~df["speaker_name"].str.contains(
            "^speaker|moderator|audio|computer|computer voice|facilitator|group|highlight|interpreter|interviewer|multiple voices|other speaker|participant|redacted|speaker X|unknown|video"
        )
    ]
    df["speaker_name"] = df["speaker_name"].apply(lambda x: re.sub(r"^\s+|\s+$", "", x))
    # kick every row out where adherence_to_guide is nan
    # df = df.dropna(subset=["adherence_to_guide"])

    selection = True
    if selection:
        features = [
            "phaticity ratio",
            "Analytic",
            "Clout",
            "Authentic",
            "Tone",
            # "WPS",
            # "duration",
            # "Rd",
            # "Rc",
            "Personal story",
            "Personal experience",
            # "QMark",
            # "Validation Strategies",
            # "Invitations to Participate",
            # "Facilitation Strategies",
            "Cognition",
            # "Social",
            "Responsivity",
            # "fac_to_part_ratio",
            # "participant_semantic_speed",
            # "facilitator_semantic_speed",
            # "role_change_rate",
            # "adherence_to_guide"
        ]
    else:
        features = df.columns.difference(
            ["Unnamed: 0", "conversation_id", "speaker_name"]
        )

    # Plot a correlation matrix
    correlation_matrix = df[features].corr()
    plt.figure(figsize=(12, 6))
    plt.title("Correlation Matrix")
    plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(features)), features, rotation=45)
    plt.yticks(range(len(features)), features)
    plt.tight_layout()
    plt.show()

    # Winsorize the data to handle outliers
    # Normalize the data to have zero mean and unit variance
    # Preserve 'conversation_id' and 'speaker_name' columns
    preserved_columns = df[["conversation_id", "speaker_name"]].reset_index(drop=True)
    df = df.drop(columns=["conversation_id", "speaker_name"]).reset_index(drop=True)
    df = df[features]
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
    df.corr()
    # print("Correlation Matrix:\n", correlation_matrix)

    # Add back the preserved columns
    df = pd.concat([preserved_columns, df], axis=1)

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
    # Ensure column names are stripped of leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Exclude the "cluster" column
    feature_columns = df.columns[df.columns != "cluster"]

    centroids_df = pd.DataFrame(centroids, columns=feature_columns)
    centroids_df.index.name = "Cluster"

    feature_means = df.mean()

    for cluster_idx in range(n_clusters):
        plt.figure(figsize=(12, 6))
        deviations = centroids_df.iloc[cluster_idx] - feature_means
        # Exclude cluster from deviations
        if "cluster" in deviations.index:
            deviations = deviations.drop("cluster")
        deviations.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.axhline(0, color="gray", linestyle="--")
        plt.title(f"Cluster {cluster_idx} Feature Deviations from Mean")
        plt.xlabel("Features")
        plt.ylabel("Deviation from Mean")
        plt.tight_layout()
        plt.show()

    return centroids_df


# -----------------------------
# New PCA Functions for Feature Analysis
# -----------------------------

def pca_analysis_overall(df, feature_columns=None, n_components=2):
    """
    Perform PCA on the dataset to determine the contribution of each feature.
    If feature_columns is not provided, it will exclude 'conversation_id', 'speaker_name', 'cluster', and any UMAP columns.
    """
    if feature_columns is None:
        exclude = {"conversation_id", "speaker_name", "cluster"}
        feature_columns = [col for col in df.columns if col not in exclude and not col.startswith("umap_")]
    
    # Standardize the features
    scaled_data = StandardScaler().fit_transform(df[feature_columns])
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create a DataFrame for the PCA scores
    pc_df = pd.DataFrame(data=principal_components, columns=[f"PC{i+1}" for i in range(n_components)])
    if "cluster" in df.columns:
        pc_df["cluster"] = df["cluster"].values
    
    explained_variance = pca.explained_variance_ratio_
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=[f"PC{i+1}" for i in range(n_components)], 
        index=feature_columns
    )
    
    print("Explained Variance Ratio:", explained_variance)
    print("PCA Loadings (feature contributions to each PC):\n", loadings)
    return pca, pc_df, loadings, explained_variance


def plot_pca_scores(pc_df):
    """Plot the first two PCA components, coloring points by cluster."""
    fig = px.scatter(
        pc_df, 
        x="PC1", 
        y="PC2", 
        color="cluster", 
        title="PCA Scatter Plot by Cluster",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}
    )
    fig.show()


# -----------------------------
# Main Script Execution
# -----------------------------
if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\facilitators_features_big.csv"
    print("Loading data")
    df = load_data(input_path)
    # df = df.dropna(subset=["Latent_Attention_Embedding"])
    # df = df[df["is_fac"] == False]

    # print("Aggregating features by speaker")
    # df = aggregate_features(df)

    # Exclude specific columns
    # exclude_columns = [
    #    "Facilitation Strategies",
    #    "Invitations to Participate",
    #    "Validation Strategies",
    # ]
    # exclude_columns = ["Rc", "Rd"]
    # df = df.drop(columns=exclude_columns)

    print("Applying UMAP to all features")
    df = preprocessing(df)

    # Exclude non-numeric columns for clustering
    df_n = df.drop(columns=["conversation_id", "speaker_name"])
    n_clusters = determine_cluster(df_n)

    print("K-means clustering")
    df = k_means(df, n_clusters)

    # --- PCA Analysis added here ---
    print("Performing PCA for feature importance analysis")
    pca, pc_df, loadings, explained_variance = pca_analysis_overall(df)
    plot_pca_scores(pc_df)
    # ---------------------------------

    df = apply_umap(df)
    plot_clusters(df)
    
    collection_ids = pd.read_csv(r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure.csv")
    df = df.merge(collection_ids[['conversation_id', 'collection_id']], on="conversation_id")
    # Save the clustered data
    output_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\facilitators_features_clustered.csv"
    df.to_csv(output_path, index=False)
