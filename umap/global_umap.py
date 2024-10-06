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
        self.aggregate_embeddings = self.config["aggregate_embeddings"]
        self.supervised_umap_enabled = self.config['supervised_umap']['enabled']
        self.supervised_umap_label_column = self.config['supervised_umap']['label_column']

        self.df = None
        self.speaker_embeddings = None
        self.load_data()

    def load_data(self):
        print(f"Loading data from {self.input_path}")
        self.df = pd.read_pickle(self.input_path)
        self.df['Latent-Attention_Embedding'] = self.df['Latent-Attention_Embedding'].apply(np.array)
        self.df.rename(columns={'Latent-Attention_Embedding': 'Latent_Attention_Embedding'}, inplace=True)
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['Latent_Attention_Embedding'])
        dropped_count = initial_count - len(self.df)
        print(f"Dropped {dropped_count} rows due to NaN values in 'Latent-Attention_Embedding'")

    def compute_umap(self, data):
        scaled_X = StandardScaler().fit_transform(np.vstack(data['Latent_Attention_Embedding'].values))

        if self.supervised_umap_enabled:
            labels = data[self.supervised_umap_label_column].values
            print(f"Using supervised UMAP with label column: {self.supervised_umap_label_column}")
            reducer = umap.UMAP(
                **self.umap_params,
                target_metric='categorical'
            )
            embedding_2d = reducer.fit_transform(scaled_X, y=labels)
        else:
            print("Using unsupervised UMAP")
            reducer = umap.UMAP(**self.umap_params)
            embedding_2d = reducer.fit_transform(scaled_X)
        
        return embedding_2d

    def compute_aggregated_embeddings(self):
        if self.aggregate_embeddings:
                self.df['speaker_name'] = self.df['speaker_name'].str.lower().str.strip()
                self.df = self.df[~self.df['speaker_name'].str.contains('interpreter|other speaker|computer voice|audio|highlight|interviewer|multiple voices|group|facilitator|video|participant|^speaker')]
                
                if self.aggregate_on_collection:
                    print("Aggregating on collection level")
                    group_columns = ['collection_id', "collection_title", "speaker_name", "is_fac"]
                else:
                    print("Aggregating on conversation level")
                    group_columns = ['conversation_id', "collection_title", "speaker_name", "is_fac"]
                
                self.speaker_embeddings = self.df.groupby(group_columns).agg(
                    Latent_Attention_Embedding=('Latent_Attention_Embedding', lambda x: np.mean(x, axis=0)),
                ).reset_index()
        else:
            print("Embedding each point without aggregation")
            self.speaker_embeddings = self.df.copy()
            self.speaker_embeddings['Wrapped_Content'] = self.speaker_embeddings['words'].apply(lambda x: '<br>'.join(textwrap.wrap(x, width=50)))

    def plot_aggregated(self):
        self.compute_aggregated_embeddings()

        embedding_2d = self.compute_umap(self.speaker_embeddings)

        self.speaker_embeddings['UMAP_1'] = embedding_2d[:, 0]
        self.speaker_embeddings['UMAP_2'] = embedding_2d[:, 1]
        self.speaker_embeddings['UMAP_3'] = embedding_2d[:, 2]

        self.speaker_embeddings['symbol'] = self.speaker_embeddings['is_fac'].apply(
            lambda is_fac: 'triangle-up' if is_fac else 'circle'
        )

        if self.show_only == "facilitators":
            self.speaker_embeddings = self.speaker_embeddings[self.speaker_embeddings['symbol'] == 'triangle-up']
        elif self.show_only == "participants":
            self.speaker_embeddings = self.speaker_embeddings[self.speaker_embeddings['symbol'] == 'circle']

        print(self.speaker_embeddings[['is_fac', 'symbol']].head())
        level = 'Collection' if self.aggregate_on_collection else 'Conversation'

        hover_data = {
            "speaker_name": True,
            "Wrapped_Content": True,
            "SpeakerTurn": True,
            'UMAP_1': False,
            'UMAP_2': False,
            'UMAP_3': False
        }

        df_sorted = self.speaker_embeddings.sort_values(by='collection_title')
        title = f'{self.model}: {"Aggregated" if self.aggregate_embeddings else "Individual"} {self.show_only.title()} Turn Embeddings for {self.collection_name} at {level} Level'

        fig = px.scatter_3d(
            df_sorted,
            x='UMAP_1',
            y='UMAP_2',
            z='UMAP_3',
            color='collection_title',
            symbol='symbol',
            title=title,
            hover_name='speaker_name',
            hover_data=hover_data,
            color_discrete_sequence=self.custom_color_palette
        )

        fig.update_traces(marker=dict(size=self.plot_marker_size, line=dict(width=self.plot_marker_line_width, color='black')))

        fig.update_layout(
            legend_title_text='Role',
            legend=dict(
                itemsizing='constant'
            )
        )

        fig.write_html(self.output_path_template)
        fig.show()
        print(f"Saved {'aggregated' if self.aggregate_embeddings else 'individual'} UMAP plot for {self.collection_name} at {level} Level (Show: {self.show_only})")


# Usage
config_path = './config.json'
visualizer = GlobalEmbeddingVisualizer(config_path)
visualizer.plot_aggregated()
