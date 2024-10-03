import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import json

class GlobalEmbeddingVisualizer:
    def __init__(self, config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

        self.model = self.config['model']
        self.collection_name = self.config['collection_name']
        self.input_path = self.config['input_path_template']
        self.output_path_template = self.config['output_path_template']
        self.custom_color_palette = self.config['custom_color_palette']
        self.umap_params = self.config['umap_params']
        self.scaler = self.config['scaler']
        self.plot_marker_size = self.config['plot_marker_size']
        self.plot_marker_line_width = self.config['plot_marker_line_width']
        self.show_only = self.config["show_only"]
        self.aggregate_on_collection = self.config["aggregate_on_collection"]
        self.df = None
        self.speaker_embeddings = None
        self.load_data()

    def load_data(self):
        print(f"Loading data from {self.input_path}")
        self.df = pd.read_pickle(self.input_path)
        self.df['Latent-Attention_Embedding'] = self.df['Latent-Attention_Embedding'].apply(np.array)
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['Latent-Attention_Embedding'])
        dropped_count = initial_count - len(self.df)
        print(f"Dropped {dropped_count} rows due to NaN values in 'Latent-Attention_Embedding'")

    def compute_aggregated_embeddings(self, aggregate_on_collection=True):
        self.df['speaker_name'] = self.df['speaker_name'].str.lower().str.strip()
        self.df = self.df[~self.df['speaker_name'].str.contains('interpreter|other speaker|computer voice|audio|highlight|interviewer|multiple voices|group|facilitator|video|participant|^speaker')]
        
        self.aggregate_on_collection = True if aggregate_on_collection == 'True' else False
        if aggregate_on_collection:
            print("Aggregating on collection level")
            group_columns = ['collection_id', 'speaker_name']
        else:
            print("Aggregating on conversation level")
            group_columns = ['conversation_id', 'speaker_name']

        self.speaker_embeddings = self.df.groupby(group_columns).agg(
            Latent_Attention_Embedding=('Latent-Attention_Embedding', lambda x: np.mean(np.vstack(x), axis=0)),
            conversation_count=('conversation_id', 'nunique'),
            is_fac=('is_fac', 'first'),
            collection_title=('collection_title', 'first')
        ).reset_index()

        if not aggregate_on_collection:
            self.speaker_embeddings['conversation_id'] = self.speaker_embeddings['conversation_id'].astype(str)
            self.df['conversation_id'] = self.df['conversation_id'].astype(str)

            conversation_info = self.df.groupby(['conversation_id', 'speaker_name']).apply(
                lambda x: {
                    'collection_title': x['collection_title'].unique()[0],
                    'collection_id': x['collection_id'].unique()[0],
                    'speaker_id': ', '.join(map(str, x['speaker_id'].unique()))
                }).reset_index()

            conversation_info[["collection_title", "collection_id", 'speaker_id']] = pd.DataFrame(
                conversation_info[0].tolist(), index=conversation_info.index)
            conversation_info = conversation_info.drop(columns=[0])
            conversation_info['conversation_id'] = conversation_info['conversation_id'].astype(str)
            conversation_info = conversation_info.drop(columns=['collection_title'])
            self.speaker_embeddings = pd.merge(self.speaker_embeddings, conversation_info, on=['conversation_id', 'speaker_name'])
        else:
            speaker_info = self.df.groupby(['collection_id', 'speaker_name']).agg(
                speaker_id=('speaker_id', lambda x: ', '.join(map(str, x.unique()))),
                conversation_ids=('conversation_id', lambda x: ', '.join(map(str, x.unique())))
            ).reset_index()
            self.speaker_embeddings = pd.merge(self.speaker_embeddings, speaker_info, on=['collection_id', 'speaker_name'])


    def compute_umap(self, data):
        scaled_X = StandardScaler().fit_transform(np.vstack(data['Latent_Attention_Embedding'].values))
        reducer = umap.UMAP(**self.umap_params)
        embedding_2d = reducer.fit_transform(scaled_X)
        return embedding_2d

    def plot_aggregated(self):
        # Compute aggregated embeddings
        self.compute_aggregated_embeddings(aggregate_on_collection=self.aggregate_on_collection)

        # Perform UMAP dimensionality reduction
        embedding_2d = self.compute_umap(self.speaker_embeddings)

        # Add UMAP components to speaker embeddings
        self.speaker_embeddings['UMAP_1'] = embedding_2d[:, 0]
        self.speaker_embeddings['UMAP_2'] = embedding_2d[:, 1]
        self.speaker_embeddings['UMAP_3'] = embedding_2d[:, 2]

        self.speaker_embeddings['collection_title'] = self.speaker_embeddings['collection_title'].str.strip()

        # Add symbols based on whether the speaker is a facilitator or participant
        self.speaker_embeddings['symbol'] = self.speaker_embeddings['is_fac'].apply(lambda x: 'triangle-up' if x else 'circle')

        # Show only facilitators or participants if requested
        if self.show_only == "facilitators":
            self.speaker_embeddings = self.speaker_embeddings[self.speaker_embeddings['symbol'] == 'triangle-up']
        elif self.show_only == "participants":
            self.speaker_embeddings = self.speaker_embeddings[self.speaker_embeddings['symbol'] == 'circle']

        level = 'Collection' if self.aggregate_on_collection else 'Conversation'

        # Hover data configuration
        hover_data = {
            "collection_title": True,
            "speaker_id": True,
            'UMAP_1': False,
            'UMAP_2': False,
            'UMAP_3': False
        }

        # Add conversation_id to hover data
        if self.aggregate_on_collection:
            hover_data["conversation_ids"] = True
        elif 'conversation_id' in self.speaker_embeddings.columns:
            hover_data["conversation_id"] = True

        # Title based on show_all config
        title = f'{self.model}: Aggregated {self.show_only.title()} Turn Embeddings for {self.collection_name} at {level} Level'

        # Plot the 3D scatter plot
        fig = px.scatter_3d(
            self.speaker_embeddings,
            x='UMAP_1',
            y='UMAP_2',
            z='UMAP_3',
            color='collection_title',
            symbol='symbol',  # Symbols added for facilitators and participants
            title=title,
            hover_name='speaker_name',
            hover_data=hover_data,
            color_discrete_sequence=self.custom_color_palette
        )

        # Update marker properties
        fig.update_traces(marker=dict(size=self.plot_marker_size, line=dict(width=self.plot_marker_line_width, color='black')))
        
        # Save and show the plot
        fig.write_html(self.output_path_template)
        fig.show()
        print(f"Saved aggregated UMAP plot for {self.collection_name} at {level} Level (Show: {self.show_only})")


# Usage
config_path = './config.json'
visualizer = GlobalEmbeddingVisualizer(config_path)
visualizer.plot_aggregated()
