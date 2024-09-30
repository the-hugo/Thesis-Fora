import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import textwrap


class GlobalEmbeddingVisualizer:
    def __init__(self, model, collection_name, input_path_template, output_path_template):
        # Instead of calling the parent class for setting the input_path, directly set it in this class
        self.model = model
        self.collection_name = collection_name
        self.input_path = input_path_template  # Directly set the input path from the argument
        self.output_path_template = output_path_template
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

    def compute_aggregated_embeddings(self):
        self.speaker_embeddings = self.df.groupby('speaker_name')['Latent-Attention_Embedding'].apply(
            lambda x: np.mean(np.vstack(x), axis=0)).reset_index()
        speaker_info = self.df[['is_fac', "speaker_name"]].drop_duplicates()
        conversation_info = self.df.groupby('speaker_name').apply(
            lambda x: {
            'SpeakerTurn': ', '.join(map(str, x['SpeakerTurn'].values)),
            'id': ', '.join(map(str, x['id'].values)),
            'speaker_id': ', '.join(map(str, x['speaker_id'].unique()))  # Assuming each speaker has a unique ID
            }).reset_index()

        conversation_info[['SpeakerTurn', 'id', 'speaker_id']] = pd.DataFrame(
        conversation_info[0].tolist(), index=conversation_info.index)
        conversation_info = conversation_info.drop(columns=[0])
        self.speaker_embeddings = pd.merge(self.speaker_embeddings, speaker_info, on='speaker_name')
        self.speaker_embeddings = pd.merge(self.speaker_embeddings, conversation_info, on='speaker_name')

    def compute_umap(self, data):
        scaled_X = StandardScaler().fit_transform(np.vstack(data['Latent-Attention_Embedding'].values))
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(scaled_X)
        return embedding_2d

    def plot_aggregated(self):
        self.compute_aggregated_embeddings()
        embedding_2d = self.compute_umap(self.speaker_embeddings)
        self.speaker_embeddings['UMAP_1'] = embedding_2d[:, 0]
        self.speaker_embeddings['UMAP_2'] = embedding_2d[:, 1]
        self.speaker_embeddings['Facilitator'] = self.speaker_embeddings['is_fac'].map(
            {True: 'Facilitator', False: 'Participant'})
        self.speaker_embeddings['SpeakerTurn'] = self.speaker_embeddings['SpeakerTurn'].apply(lambda x: '<br>'.join(textwrap.wrap(x, width=50)))
        self.speaker_embeddings['speaker_id'] = self.speaker_embeddings['speaker_id'].apply(lambda x: '<br>'.join(textwrap.wrap(x, width=50)))
        fig = px.scatter(   
            self.speaker_embeddings,
            x='UMAP_1', 
            y='UMAP_2',
            color='Facilitator',
            title=f'{self.model}: Aggregated Speaker Turn Embeddings for {self.collection_name}',
            #labels={'UMAP_1': 'UMAP Dimension 1', 'UMAP_2': 'UMAP Dimension 2'},
            hover_name='speaker_name',
            hover_data={'SpeakerTurn': True,
                        "speaker_id": True,
                        #"Turn Distribution": True,
                            'Facilitator': False,
                            'UMAP_1': False,            
                            'UMAP_2': False},
            color_discrete_map={'Facilitator': 'red', 'Participant': 'blue'}
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='black')), selector=dict(name='Facilitator'))
        fig.write_html(self.output_path_template)
        print(f"Saved aggregated UMAP plot for {self.collection_name}")

model = "nv-embed"
collection_name = "corpus_data"
input_path_template = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\output\umap\data_nv-embed_processed_output.pkl"
output_path_template = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\output\graphs\nv-embed\corpus\global_umap.html"

visualizer = GlobalEmbeddingVisualizer(model, collection_name, input_path_template, output_path_template)
visualizer.plot_aggregated()
