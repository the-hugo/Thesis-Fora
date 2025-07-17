import os
import json
import numpy as np
import pandas as pd
import umap
import plotly.express as px
from sklearn.preprocessing import StandardScaler

def animate_conversation(convo_df, conv_id):
    """
    Compute a UMAP embedding for a single conversation and create an animated
    scatter plot showing the conversation trajectory through semantic space.
    Each frame shows all turns up to that point.
    """
    # Stack embeddings and scale them.
    embeddings = np.vstack(convo_df["Latent_Attention_Embedding"].values)
    scaled_embeddings = StandardScaler().fit_transform(embeddings)
    
    # Create a UMAP reducer with fixed parameters.
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_embedding = reducer.fit_transform(scaled_embeddings)
    
    # Assign UMAP coordinates to the DataFrame.
    convo_df = convo_df.copy()  # avoid modifying the original
    convo_df["UMAP_1"] = umap_embedding[:, 0]
    convo_df["UMAP_2"] = umap_embedding[:, 1]
    
    # Ensure there is a turn ordering column.
    if "SpeakerTurn" not in convo_df.columns:
        convo_df["SpeakerTurn"] = np.arange(1, len(convo_df) + 1)
    convo_df = convo_df.sort_values("SpeakerTurn")
    
    # Create cumulative data so that each animation frame includes all past turns.
    cumulative_frames = []
    for turn in convo_df["SpeakerTurn"].unique():
        cum_df = convo_df[convo_df["SpeakerTurn"] <= turn].copy()
        cum_df["Frame"] = turn  # designate the current frame number
        cumulative_frames.append(cum_df)
    anim_df = pd.concat(cumulative_frames)
    
    # Create an animated scatter plot.
    fig = px.scatter(
        anim_df,
        x="UMAP_1",
        y="UMAP_2",
        animation_frame="Frame",
        hover_data=["SpeakerTurn"],
        title=f"Semantic Trajectory for Conversation {conv_id}"
    )
    
    # Clean up the layout.
    fig.update_layout(
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Save the animation as an interactive HTML file.
    output_filename = f"conversation_{conv_id}_animation.html"
    fig.write_html(output_filename)
    print(f"Saved animated plot for conversation {conv_id} as {output_filename}")
    fig.show()

def main():
    # Load your data.
    # Ensure that your data file (a pickle) is in the same folder or adjust the path accordingly.
    data_path = r"C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/output/embeddings/data_nv-embed_processed_output.pkl"  # change this if your data is located elsewhere
    
    df = pd.read_pickle(data_path)
    
    # Ensure the embeddings are proper numpy arrays.
    df["Latent_Attention_Embedding"] = df["Latent-Attention_Embedding"].apply(np.array)
    
        # Select 10 unique conversation IDs.
    unique_conv_ids = df["conversation_id"].unique()[:10]
    print(f"Processing {len(unique_conv_ids)} conversations.")
    
    for conv_id in unique_conv_ids:
        convo_df = df[df["conversation_id"] == conv_id]
        
        # Skip conversations with very few turns.
        if len(convo_df) < 2:
            print(f"Skipping conversation {conv_id} due to insufficient turns.")
            continue
        
        animate_conversation(convo_df, conv_id)
    
if __name__ == "__main__":
    main()