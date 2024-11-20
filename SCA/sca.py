from semantic_components.sca import SCA
import pandas as pd
import numpy as np
import sys

def load_data(input_path):
    print(f"Loading data from {input_path}")
    df = pd.read_pickle(input_path)
    return df


if __name__ == "__main__":

    debug = False
    if len(sys.argv) == 2:
        debug = sys.argv[1] == "debug"

    input_path = "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/output/embeddings/data_nv-embed_processed_output.pkl"
    output_path = "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/output/SCA/SCA_data_nv-embed_processed_output.pkl"

    df = load_data(input_path)
    df = df.dropna(subset=["Latent-Attention_Embedding"])

    # filter samples with less than 20 chars
    df_sampled = df[df["words"].str.len() >= 20]

    # groupby speaker name and conversation id and concatenate all words
    
    df_sampled = df.groupby(["speaker_name", "conversation_id"]).agg({
        "words": lambda x: " ".join(x),
        "Latent-Attention_Embedding": lambda x: np.mean(np.vstack(x), axis=0)
    }).reset_index()
    
    print(len(df))

    if debug:
        df_sampled = df_sampled.sample(10000)
    
    df_sampled["text"] = df_sampled["words"]
    embeddings = np.vstack(df_sampled["Latent-Attention_Embedding"].values)
    
    sca = SCA(alpha_decomposition=0.1, mu=0.9, combine_overlap_threshold=0.5, max_iterations=5, verbose=True)
    scores, residuals, ids = sca.fit(df_sampled, embeddings)

    # get representations and explainable transformations
    representations = sca.representations  # pandas df
    transformed = sca.transform(embeddings)  # equivalent to variable scores above
    
    # merge representations with original data key: representation_medoid on df["words"]. The rest leave empty
    df = df.merge(representations, left_on="words", right_on="representation_medoid", how="left")
    representations.to_pickle("representations.pkl")
    
    print(representations)