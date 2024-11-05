import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_data(input_path):
    """Load data from a pickle file."""
    print(f"Loading data from {input_path}")
    return pd.read_pickle(input_path)


df = load_data(
    "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/output/annotated/data_llama70B_processed_output_phatic_ratio.pkl_temp_24000_phatic_ratio.pkl"
)
"""
df = load_data(
    "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/output/annotated/output_filled_phatic_ratio_conversation_speaker_phatic_ratio.pkl"
)
"""
# data/output/annotated/
# print all columns
print(df.columns)

# drop Unnamed: 0, id, source_type, normalized time

df.drop(
    columns=[
        "Unnamed: 0",
        "id",
        "source_type",
        #"normalized_time",
        "collection_title",
        "annotated",
        "location",
        "words",
        "speaker_name",
        "start_time",
        "phatic speech",
        "Latent-Attention_Embedding",
    ],
    inplace=True,
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
"""
"""
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
"""
# calculate the correlation matrix
correlation_matrix = df.corr()

# print the correlation matrix
print(correlation_matrix)

# plot the correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True)
plt.show()
