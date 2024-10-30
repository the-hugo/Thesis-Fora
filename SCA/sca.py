from semantic_components.sca import SCA
import pandas as pd
import numpy as np

def load_data(input_path):
    print(f"Loading data from {input_path}")
    df = pd.read_pickle(input_path)
    return df


if __name__ == "__main__":
    input_path = "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/output/embeddings/data_nv-embed_processed_output.pkl"
    output_path = "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/output/SCA/SCA_data_nv-embed_processed_output.pkl"

    df = load_data(input_path)
    df = df.dropna(subset=["Latent-Attention_Embedding"])
    
    df["text"] = df["words"]
    embeddings = np.vstack(df["Latent-Attention_Embedding"].values)
    
    print(type(df["text"]))
    print(type(embeddings))
    
    sca = SCA(alpha_decomposition=0.1, mu=0.9, combine_overlap_threshold=0.5)
    scores, residuals, ids = sca.fit(df, embeddings)

    # get representations and explainable transformations
    representations = sca.representations  # pandas df
    transformed = sca.transform(embeddings)  # equivalent to variable scores above
    
    # merge representations with original data
    df = pd.merge(df, representations, left_index=True, right_index=True)
    representations.to_pickle("representations.pkl")
    
    print(representations)