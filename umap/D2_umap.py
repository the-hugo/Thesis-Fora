import umap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import textwrap


class EmbeddingVisualizer:
    def __init__(self, model, input_path_template, output_path_template):
        self.model = model
        self.input_path = input_path_template
        self.output_path_template = output_path_template
        self.df = None
        self.speaker_embeddings = None
        self.load_data()

    def load_data(self):
        print(f"Loading data from {self.input_path}")
        self.df = pd.read_pickle(self.input_path)
        self.df["Latent-Attention_Embedding"] = self.df[
            "Latent-Attention_Embedding"
        ].apply(np.array)
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=["Latent-Attention_Embedding"])
        dropped_count = initial_count - len(self.df)
        print(
            f"Dropped {dropped_count} rows due to NaN values in 'Latent-Attention_Embedding'"
        )

    def compute_aggregated_embeddings(self):
        self.speaker_embeddings = (
            self.df.groupby("speaker_name")["Latent-Attention_Embedding"]
            .apply(lambda x: np.mean(np.vstack(x), axis=0))
            .reset_index()
        )
        speaker_info = self.df[["is_fac", "speaker_name"]].drop_duplicates()
        conversation_info = (
            self.df.groupby("speaker_name")
            .apply(
                lambda x: {
                    "Index in Conversation": ", ".join(
                        map(str, x["SpeakerTurn"].values)
                    ),
                    #'Snippet ID': ', '.join(map(str, x['Snippet ID'].values)),
                    "Speaker ID": ", ".join(
                        map(str, x["speaker_id"].unique())
                    ),  # Assuming each speaker has a unique ID
                }
            )
            .reset_index()
        )

        # self.speaker_embeddings['Unique Speaker Turns'] = self.df.groupby('Speaker Name')['Snippet ID'].nunique().values
        # self.speaker_embeddings['Average Turn Length'] = self.df.groupby('Speaker Name')['duration'].mean().values
        conversation_info[["Index in Conversation", "Speaker ID"]] = pd.DataFrame(
            conversation_info[0].tolist(), index=conversation_info.index
        )
        conversation_info = conversation_info.drop(columns=[0])

        self.speaker_embeddings = pd.merge(
            self.speaker_embeddings, speaker_info, on="speaker_name"
        )
        self.speaker_embeddings = pd.merge(
            self.speaker_embeddings, conversation_info, on="speaker_name"
        )

    def compute_umap(self, data):
        scaled_X = StandardScaler().fit_transform(
            np.vstack(data["Latent-Attention_Embedding"].values)
        )
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(scaled_X)
        return embedding_2d

    def plot_aggregated(self):
        self.compute_aggregated_embeddings()
        embedding_2d = self.compute_umap(self.speaker_embeddings)
        self.speaker_embeddings["UMAP_1"] = embedding_2d[:, 0]
        self.speaker_embeddings["UMAP_2"] = embedding_2d[:, 1]
        self.speaker_embeddings["Facilitator"] = self.speaker_embeddings["is_fac"].map(
            {True: "Facilitator", False: "Participant"}
        )

        self.speaker_embeddings["Index in Conversation"] = self.speaker_embeddings[
            "SpeakerTurn"
        ].apply(lambda x: "<br>".join(textwrap.wrap(x, width=50)))
        self.speaker_embeddings["Speaker ID"] = self.speaker_embeddings[
            "speaker_id"
        ].apply(lambda x: "<br>".join(textwrap.wrap(x, width=50)))
        fig = px.scatter(
            self.speaker_embeddings,
            x="UMAP_1",
            y="UMAP_2",
            color="Facilitator",
            title=f"{self.model}: Aggregated Speaker Turn Embeddings for {self.collection_name}",
            # labels={'UMAP_1': 'UMAP Dimension 1', 'UMAP_2': 'UMAP Dimension 2'},
            hover_name="Speaker Name",
            hover_data={
                "Index in Conversation": True,
                # "Average Turn Length": True,
                # "Unique Speaker Turns": True,
                "Speaker ID": True,
                # "Turn Distribution": True,
                "Facilitator": False,
                "UMAP_1": False,
                "UMAP_2": False,
            },
            color_discrete_map={"Facilitator": "red", "Participant": "blue"},
        )
        fig.update_traces(
            marker=dict(size=10, line=dict(width=2, color="black")),
            selector=dict(name="Facilitator"),
        )
        fig.write_html(
            self.output_path_template.format(
                self.model, self.collection_name, "collection_aggregated_umap"
            )
        )
        print(f"Saved aggregated UMAP plot for {self.collection_name}")

    def plot_by_conversation(self):
        """
        Use this chunk of code to filter for particular speakers and collections

        collections = ['United Way of Dane County']
        speakers = ['Mathias']

        # Filter conversation_ids that include the specified speakers
        conversation_ids_with_speakers = self.df[self.df['speaker_name'].isin(speakers)]['conversation_id'].unique()

        # Filter conversation_ids that are part of the specified collections
        conversation_ids_in_collections = self.df[self.df['collection_title'].isin(collections)]['conversation_id'].unique()

        # Keep rows where conversation_id is in both filtered lists
        valid_conversation_ids = np.intersect1d(conversation_ids_with_speakers, conversation_ids_in_collections)
        self.df = self.df[self.df['conversation_id'].isin(valid_conversation_ids)]
        """
        # please only plot conversation_ids Top 10
        valid_conversation_ids = [
            871, 2061, 804, 1740, 654, 910, 2261, 581, 2089, 524,
            2179, 812, 870, 720, 919, 2409, 787, 2358, 2188, 1103
        ]
        self.df = self.df[self.df['conversation_id'].isin(valid_conversation_ids)]

        for conversation_id, group in self.df.groupby("conversation_id"):
            embedding_2d = self.compute_umap(group)
            group["UMAP_1"] = embedding_2d[:, 0]
            group["UMAP_2"] = embedding_2d[:, 1]
            group["Facilitator"] = group["is_fac"].map(
                {True: "Facilitator", False: "Non-Facilitator"}
            )
            group["Wrapped Content"] = group["words"].apply(
                lambda x: "<br>".join(textwrap.wrap(x, width=50))
            )
            group["duration"] = group["duration"].astype(int)
            collection_name = group["collection_title"].iloc[0]
            fig = px.scatter(
                group,
                x="UMAP_1",
                y="UMAP_2",
                color="Facilitator",
                title=f"{self.model}: Embedding for Conversation {conversation_id} in {collection_name}",
                # labels={'Words': 'Content', "Duration": "duration"},
                hover_name="speaker_name",
                hover_data={
                    "SpeakerTurn": True,
                    "Wrapped Content": True,
                    "duration": True,
                    "Facilitator": False,
                    "UMAP_1": False,
                    "UMAP_2": False,
                },
                color_discrete_map={
                    "Facilitator": "#ffc600",
                    "Non-Facilitator": "#00a4eb",
                },
            )
            if "&" in collection_name:
                collection_name = collection_name.replace("&", "and").replace(":", " ")
            fig.update_traces(
                marker=dict(size=10, line=dict(width=2, color="black")),
                selector=dict(name="Facilitator"),
            )
            final_output_path = (
                self.output_path_template
                + "/"
                + str(collection_name)
                + "_"
                + str(conversation_id)
                + ".html"
            )
            fig.write_html(final_output_path)
            fig.show()
            print(
                f"Saved UMAP plot for Conversation {conversation_id} in {collection_name}"
            )


# Usage example
model = "nv-embed"
input_path_template = "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/output/embeddings/data_nv-embed_processed_output.pkl"
output_path_template = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\graphs\nv-embed\corpus\conversations"

visualizer = EmbeddingVisualizer(model, input_path_template, output_path_template)
# visualizer.plot_aggregated()
visualizer.plot_by_conversation()
