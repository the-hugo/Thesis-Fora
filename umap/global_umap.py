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

        self.plot_mode = self.config["plot_mode"]
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
        conversation_info[conversation_info_columns] = pd.DataFrame(
            conversation_info[0].tolist(), index=conversation_info.index
        )
        conversation_info = conversation_info.drop(columns=[0])
        self.convo_info = {col: True for col in conversation_info_columns}
        return conversation_info

    def compute_aggregated_embeddings(self):
        self.df["speaker_name"] = self.df["speaker_name"].str.strip()
        self.df = self.df[
            ~self.df["speaker_name"].str.contains(
                "^speaker|moderator|audio|computer|computer voice|facilitator|group|highlight|interpreter|interviewer|multiple voices|other speaker|participant|redacted|speaker X|unknown|video"
            )
        ]

        # assign labels to the roles
        symbols = {"Facilitator": "square", "Participant": "circle"}
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
        self.compute_aggregated_embeddings()

        # Compute UMAP embedding and add columns.
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
        }

        if self.color_by_role == "f_p":
            self.speaker_embeddings["symbol"] = self.speaker_embeddings["is_fac"].apply(
                lambda x: "Facilitator" if x else "Participant"
            )
            color_column = "symbol"
            custom_color_palette = ["#ffc600", "#00a4eb"]
            legend_title = "Role"
            symbol_sequence = ["diamond", "circle"]
        else:
            color_column = "collection_title"
            custom_color_palette = self.custom_color_palette
            legend_title = "Collection"
            symbol_sequence = None

        df_sorted = self.speaker_embeddings.sort_values(by=["symbol", "collection_title"])
        neighbors = self.umap_params["n_neighbors"]
        neighbors_str = str(neighbors)

        if hasattr(self, "plot_mode") and self.plot_mode == "2D":
            fig = px.scatter(
                df_sorted,
                x="UMAP_1",
                y="UMAP_2",
                color=color_column,
                symbol="symbol" if symbol_sequence else None,
                hover_name="speaker_name",
                hover_data=hover_data,
                color_discrete_sequence=custom_color_palette,
            )
            fig.update_layout(
                xaxis=dict(visible=False, showgrid=False, zeroline=False),
                yaxis=dict(visible=False, showgrid=False, zeroline=False),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                title="",
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
            )
            fig.update_traces(marker=dict(size=8))
        else:
            title = "Aggregated Turn Embeddings on Conversational Level"
            if symbol_sequence:
                fig = px.scatter_3d(
                    df_sorted,
                    x="UMAP_1",
                    y="UMAP_2",
                    z="UMAP_3",
                    color=color_column,
                    symbol="symbol",
                    symbol_sequence=symbol_sequence,
                    title=title,
                    hover_name="speaker_name",
                    hover_data=hover_data,
                    color_discrete_sequence=custom_color_palette,
                )
            else:
                fig = px.scatter_3d(
                    df_sorted,
                    x="UMAP_1",
                    y="UMAP_2",
                    z="UMAP_3",
                    color=color_column,
                    symbol="symbol",
                    title=title,
                    hover_name="speaker_name",
                    hover_data=hover_data,
                    color_discrete_sequence=custom_color_palette,
                )

            fig.update_traces(
                marker=dict(
                    size=self.plot_marker_size,
                    line=dict(width=self.plot_marker_line_width, color="black"),
                )
            )
            fig.update_layout(
                scene=dict(
                    xaxis=dict(visible=False, showgrid=False, zeroline=False),
                    yaxis=dict(visible=False, showgrid=False, zeroline=False),
                    zaxis=dict(visible=False, showgrid=False, zeroline=False)
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                title="",
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
            )
            fig.update_traces(marker=dict(size=8))
            fig.update_layout(
                legend_title_text=legend_title, legend=dict(itemsizing="constant")
            )
            fig.update_layout(
                font=dict(size=18),
                title=dict(x=0.5, xanchor="center"),
                legend=dict(x=0.85, y=0.5, xanchor="center"),
            )
        final_output_path = (
            self.output_path_template
            + "umap_embeddings"
            + ("_2d" if hasattr(self, "plot_mode") and self.plot_mode == "2D" else "")
            + "_"
            + level
            + "_"
            + self.show_only
            + "_"
            + neighbors_str
            + ".png"
        )
        fig.write_image(final_output_path)
        fig.show()
        print(
            f"Saved {'aggregated' if self.aggregate_embeddings else 'individual'} UMAP plot for {self.collection_name} at {level} Level (Show: {self.show_only})"
        )

    def plot_per_conversation(self):
        """
        New method: Generate a separate UMAP plot for every conversation_id.
        The plot title is set to "Embeddings for Conversation XX" (centered).
        """
        self.compute_aggregated_embeddings()

        # Set up hover data and color/symbol configuration.
        hover_data = {
            **self.convo_info,
            "UMAP_1": False,
            "UMAP_2": False,
            "UMAP_3": False,
        }

        if self.color_by_role == "f_p":
            self.speaker_embeddings["symbol"] = self.speaker_embeddings["is_fac"].apply(
                lambda x: "Facilitator" if x else "Participant"
            )
            color_column = "symbol"  # Use role labels for both color and symbol.
            custom_color_palette = ["#ffc600", "#00a4eb"]
            legend_title = "Role"
            symbol_sequence = ["diamond", "circle"]
        else:
            color_column = "collection_title"
            custom_color_palette = self.custom_color_palette
            legend_title = "Collection"
            symbol_sequence = None

        # Get unique conversation ids.
        conversation_ids = self.speaker_embeddings["conversation_id"].unique()

        for convo in conversation_ids:
            df_convo = self.speaker_embeddings[self.speaker_embeddings["conversation_id"] == convo].copy()
            if df_convo.empty:
                continue

            # Compute UMAP for the current conversation.
            embedding_2d = self.compute_umap(df_convo)
            df_convo["UMAP_1"] = embedding_2d[:, 0]
            df_convo["UMAP_2"] = embedding_2d[:, 1]
            if not (hasattr(self, "plot_mode") and self.plot_mode == "2D"):
                df_convo["UMAP_3"] = embedding_2d[:, 2]

            df_sorted = df_convo.sort_values(by=["symbol", "collection_title"])
            neighbors = self.umap_params["n_neighbors"]
            neighbors_str = str(neighbors)

            # Create plot (2D or 3D) with title "Embeddings for Conversation XX"
            if hasattr(self, "plot_mode") and self.plot_mode == "2D":
                fig = px.scatter(
                    df_sorted,
                    x="UMAP_1",
                    y="UMAP_2",
                    color=color_column,
                    symbol="symbol" if symbol_sequence else None,
                    hover_name="speaker_name",
                    hover_data=hover_data,
                    color_discrete_sequence=custom_color_palette,
                )
                fig.update_layout(
                    xaxis=dict(visible=False, showgrid=False, zeroline=False),
                    yaxis=dict(visible=False, showgrid=False, zeroline=False),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    title=f"Embeddings for Conversation {convo}",
                    title_x=0.5,
                    title_y=0.9,  # Lower the title vertically
                    margin=dict(l=0, r=0, t=50, b=0),
                    showlegend=False,
                )
                fig.update_traces(marker=dict(size=8))
            else:
                title = f"Embeddings for Conversation {convo}"
                if symbol_sequence:
                    fig = px.scatter_3d(
                        df_sorted,
                        x="UMAP_1",
                        y="UMAP_2",
                        z="UMAP_3",
                        color=color_column,
                        symbol="symbol",
                        symbol_sequence=symbol_sequence,
                        hover_name="speaker_name",
                        hover_data=hover_data,
                        color_discrete_sequence=custom_color_palette,
                        title=title,
                    )
                else:
                    fig = px.scatter_3d(
                        df_sorted,
                        x="UMAP_1",
                        y="UMAP_2",
                        z="UMAP_3",
                        color=color_column,
                        symbol="symbol",
                        hover_name="speaker_name",
                        hover_data=hover_data,
                        color_discrete_sequence=custom_color_palette,
                        title=title,
                    )
                fig.update_traces(
                    marker=dict(
                        size=self.plot_marker_size,
                        line=dict(width=self.plot_marker_line_width, color="black"),
                    )
                )
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(visible=False, showgrid=False, zeroline=False),
                        yaxis=dict(visible=False, showgrid=False, zeroline=False),
                        zaxis=dict(visible=False, showgrid=False, zeroline=False)
                    ),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=50, b=0),
                    showlegend=False,
                    title=dict(text=title, x=0.5, y=0.9, xanchor="center"),  # Adjust title position here
                )
                fig.update_traces(marker=dict(size=8))
                fig.update_layout(
                    legend_title_text=legend_title, legend=dict(itemsizing="constant")
                )
                fig.update_layout(
                    font=dict(size=18),
                )
            final_output_path = (
                self.output_path_template
                + "umap_embeddings_conversation_"
                + str(convo)
                + ("_2d" if hasattr(self, "plot_mode") and self.plot_mode == "2D" else "")
                + "_"
                + self.show_only
                + "_"
                + neighbors_str
                + ".png"
            )
            fig.write_image(final_output_path)
            fig.show()
            print(f"Saved UMAP plot for conversation {convo} at {final_output_path}")


# Usage
config_path = "./config.json"
visualizer = GlobalEmbeddingVisualizer(config_path)
visualizer.plot_aggregated()  # Existing aggregated plot
visualizer.plot_per_conversation()  # New: one plot per conversation_id
