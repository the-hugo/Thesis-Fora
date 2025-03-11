import umap
import json
import os
import numpy as np
import pandas as pd
import textwrap
import plotly.express as px
from sklearn.preprocessing import StandardScaler


class GlobalEmbeddingVisualizer:
    def __init__(self, config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)

        self.color_by_role = self.config["color_by_role"]
        self.model = self.config["model"]
        self.collection_name = self.config["collection_name"]
        self.input_path = self.config["input_path_template"]
        self.output_path_template = self.config["output_path_template"]
        self.custom_color_palette = self.config["custom_color_palette"]
        self.umap_params = self.config["umap_params"]
        self.scaler = self.config["scaler"]
        self.plot_marker_size = self.config["plot_marker_size"]
        self.plot_marker_line_width = self.config["plot_marker_line_width"]
        self.show_only = self.config["show_only"]
        self.aggregate_on_collection = self.config["aggregate_on_collection"]
        self.aggregate_embeddings = self.config["aggregate_embeddings"]
        self.supervised_umap_enabled = self.config["supervised_umap"]["enabled"]
        self.supervised_umap_label_column = self.config["supervised_umap"][
            "label_column"
        ]
        self.truncate_turns = self.config["truncate_turns"]

        self.df = None
        self.speaker_embeddings = None
        self.convo_info = None
        self.load_data()

    def load_data(self):
        print(f"Loading data from {self.input_path}")
        self.df = pd.read_pickle(self.input_path)
        # check if the column name is not Latent-Attention_Embedding
        if "Latent-Attention_Embedding" not in self.df.columns:
            # rename the column to Latent-Attention_Embedding
            self.df.rename(columns={"Latent_Attention_Embedding": "Latent-Attention_Embedding"}, inplace=True)
        self.df["Latent-Attention_Embedding"] = self.df[
            "Latent-Attention_Embedding"
        ].apply(np.array)
        self.df.rename(
            columns={"Latent-Attention_Embedding": "Latent_Attention_Embedding"},
            inplace=True,
        )
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=["Latent_Attention_Embedding"])
        dropped_count = initial_count - len(self.df)
        print(
            f"Dropped {dropped_count} rows due to NaN values in 'Latent-Attention_Embedding'"
        )
        #self.df = self.df.sample(frac=0.1)

    def compute_umap(self, data):
        scaled_X = StandardScaler().fit_transform(
            np.vstack(data["Latent_Attention_Embedding"].values)
        )

        if self.supervised_umap_enabled:
            labels = data[self.supervised_umap_label_column].values
            print(
                f"Using supervised UMAP with label column: {self.supervised_umap_label_column}"
            )
            reducer = umap.UMAP(**self.umap_params, target_metric="categorical")
            embedding_2d = reducer.fit_transform(scaled_X, y=labels)
        else:
            print("Using unsupervised UMAP")
            reducer = umap.UMAP(**self.umap_params)
            embedding_2d = reducer.fit_transform(scaled_X)

        return embedding_2d

    def truncate_quartiles(self):
        # group by conversation_id
        self.df["SpeakerTurn"] = self.df.groupby("conversation_id")["SpeakerTurn"].rank(
            method="dense", ascending=True
        )
        # cut off the first quartile and the last quartile
        self.df = self.df[
            (self.df["SpeakerTurn"] > self.df["SpeakerTurn"].quantile(0.40))
            & (self.df["SpeakerTurn"] < self.df["SpeakerTurn"].quantile(0.60))
        ]

    def create_conversation_info(self, group_columns):
        if self.aggregate_on_collection:
            conversation_info_columns = [
                "collection_title",
                "symbol",
                "is_fac",
                "conversation_ids",
                "speaker_id",
                "speaker_name",
            ]
            conversation_info = (
                self.df.groupby(group_columns)
                .apply(
                    lambda x: {
                        "collection_title": x["collection_title"].unique()[0],
                        "symbol": x["symbol"].unique()[0],
                        "is_fac": x["is_fac"].unique()[0],
                        "conversation_ids": ", ".join(
                            map(str, x["conversation_id"].unique())
                        ),
                        "speaker_id": ", ".join(map(str, x["speaker_id"].unique())),
                        "speaker_name": ", ".join(map(str, x["speaker_name"].unique())),
                    }
                )
                .reset_index()
            )
        else:
            conversation_info_columns = [
                "collection_title",
                "symbol",
                "is_fac",
                "speaker_id",
                "speaker_name",
            ]
            conversation_info = (
                self.df.groupby(group_columns)
                .apply(
                    lambda x: {
                        "collection_title": x["collection_title"].unique()[0],
                        "symbol": x["symbol"].unique()[0],
                        "is_fac": x["is_fac"].unique()[0],
                        "speaker_id": ", ".join(map(str, x["speaker_id"].unique())),
                        "speaker_name": ", ".join(map(str, x["speaker_name"].unique())),
                    }
                )
                .reset_index()
            )
        # self.speaker_embeddings["conversation_id"] = self.speaker_embeddings["conversation_id"].astype(str)
        conversation_info[conversation_info_columns] = pd.DataFrame(
            conversation_info[0].tolist(), index=conversation_info.index
        )
        conversation_info = conversation_info.drop(columns=[0])
        self.convo_info = {col: True for col in conversation_info_columns}
        return conversation_info

    def compute_aggregated_embeddings(self):
        self.df["speaker_name"] = self.df["speaker_name"].str.lower().str.strip()
        self.df = self.df[
            ~self.df["speaker_name"].str.contains(
                "^speaker|moderator|audio|computer|computer voice|facilitator|group|highlight|interpreter|interviewer|multiple voices|other speaker|participant|redacted|speaker X|unknown|video"
            )
        ]

        # assign labels to the roles
        symbols = {"Facilitator": "diamond", "Participant": "circle"}
        self.df["role"] = self.df["is_fac"].map(
            {True: "Facilitator", False: "Participant"}
        )
        self.df["symbol"] = self.df["role"].map(symbols)

        if self.truncate_turns:
            self.truncate_quartiles()

        if self.show_only == "facilitators":
            self.df = self.df[self.df["is_fac"] == True]

        elif self.show_only == "participants":
            self.df = self.df[self.df["is_fac"] == False]

        if self.aggregate_embeddings:
            if self.aggregate_on_collection:
                print("Aggregating on collection level")
                group_columns = [
                    "collection_id",
                    "speaker_name",
                ]
                conversation_info = self.create_conversation_info(group_columns)
            else:
                print("Aggregating on conversation level")
                group_columns = ["conversation_id", "speaker_name"]
                conversation_info = self.create_conversation_info(group_columns)

            self.speaker_embeddings = (
                self.df.groupby(group_columns)
                .agg(
                    Latent_Attention_Embedding=(
                        "Latent_Attention_Embedding",
                        lambda x: np.mean(x, axis=0),
                    ),
                )
                .reset_index()
            )

        else:
            print("Embedding each point without aggregation")
            self.speaker_embeddings = self.df.copy()
            self.speaker_embeddings["Wrapped_Content"] = self.speaker_embeddings[
                "words"
            ].apply(lambda x: "<br>".join(textwrap.wrap(x, width=50)))
            self.convo_info = {
                "Wrapped_Content": True,
                "SpeakerTurn": True,
                "conversation_id": True,
            }

        if self.aggregate_embeddings:
            self.speaker_embeddings = pd.merge(
                self.speaker_embeddings, conversation_info, on=group_columns
            )


    def plot_aggregated(self):
        """
        Phatic Speech Part (Continuous)
        """
        
        # Remove rows with NaN in phaticity ratio.
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=["phaticity ratio"])
        
        dropped_count = initial_count - len(self.df)
        print(f"Dropped {dropped_count} rows due to NaN values in 'phaticity ratio'")
        
        # Compute embeddings and UMAP.
        self.compute_aggregated_embeddings()
        embedding_2d = self.compute_umap(self.speaker_embeddings)

        self.speaker_embeddings["UMAP_1"] = embedding_2d[:, 0]
        self.speaker_embeddings["UMAP_2"] = embedding_2d[:, 1]
        self.speaker_embeddings["UMAP_3"] = embedding_2d[:, 2]

        if self.aggregate_embeddings:
            level = "Collection" if self.aggregate_on_collection else "Conversation"
        else:
            level = "SpeakerTurn"

        hover_data = {
            **self.convo_info,
            "UMAP_1": False,
            "UMAP_2": False,
            "UMAP_3": False,
            "phaticity ratio": True,  # display the continuous value in hover
        }

        # Sort dataframe for consistent ordering
        df_sorted = self.speaker_embeddings.sort_values(by=["symbol", "collection_title"])

        # Create a 3D scatter plot with a continuous color scale based on phaticity ratio.
        fig = px.scatter_3d(
            df_sorted,
            x="UMAP_1",
            y="UMAP_2",
            z="UMAP_3",
            color="phaticity ratio",
            hover_name="speaker_name",
            hover_data=hover_data,
            color_continuous_scale=["#ffc600", "#00B142"],
            range_color=[0, 1]
        )
        
        # Remove the color scale (colorbar) by disabling it on the trace markers and color axes.
        fig.update_traces(marker=dict(showscale=False))
        fig.update_coloraxes(showscale=False)
        
        # Update marker properties: use circles and set the desired size and outline.
        fig.update_traces(
            marker=dict(
                size=self.plot_marker_size,
                line=dict(width=self.plot_marker_line_width, color="black"),
                symbol="circle",
            )
        )

        # Compute bounds for the UMAP coordinates.
        x_min, x_max = df_sorted["UMAP_1"].min(), df_sorted["UMAP_1"].max()
        y_min, y_max = df_sorted["UMAP_2"].min(), df_sorted["UMAP_2"].max()
        z_min, z_max = df_sorted["UMAP_3"].min(), df_sorted["UMAP_3"].max()
        center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        distance_factor = 2.5  # adjustable camera distance factor
        eye = {
            "x": center[0] + distance_factor * max_range,
            "y": center[1] + distance_factor * max_range,
            "z": center[2] + distance_factor * max_range,
        }

        # Update layout: remove axis labels, gridlines, and legends.
        fig.update_layout(
            autosize=False,
            width=2560,
            height=1440,
            paper_bgcolor="rgba(0,0,0,0)",  # transparent background
            plot_bgcolor="rgba(0,0,0,0)",     # transparent plot area
            title="",                       # remove title
            scene=dict(
                camera=dict(eye=eye, projection=dict(type="orthographic")),
                xaxis=dict(
                    title="",
                    visible=False,
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    showbackground=False,
                ),
                yaxis=dict(
                    title="",
                    visible=False,
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    showbackground=False,
                ),
                zaxis=dict(
                    title="",
                    visible=False,
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    showbackground=False,
                ),
            ),
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        )
        
        fig.show()
        fig.write_html(r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\phatic speech\phaticity_umap.html")
        print(f"Saved {'aggregated' if self.aggregate_embeddings else 'individual'} UMAP plot for {self.collection_name} at {level} Level (Show: {self.show_only})")




# Usage
config_path = "./config.json"
visualizer = GlobalEmbeddingVisualizer(config_path)
visualizer.plot_aggregated()
