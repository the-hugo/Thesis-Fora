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
        # Filter the dataframe to only include rows where 'is_fac' is True
        self.df = self.df[self.df['is_fac'] == True]

        self.df['speaker_name'] = self.df['speaker_name'].str.lower().str.strip()

        # Group by 'collection_id' and 'speaker_name' and aggregate the embeddings
        self.speaker_embeddings = self.df.groupby(['collection_id', 'speaker_name']).agg(
            Latent_Attention_Embedding=('Latent-Attention_Embedding', lambda x: np.mean(np.vstack(x), axis=0)),
            conversation_count=('conversation_id', 'nunique')
        ).reset_index()

        # Conversation info aggregated per speaker name and collection_id
        conversation_info = self.df.groupby(['collection_id', 'speaker_name']).apply(
            lambda x: {
            'collection_title': x['collection_title'].unique()[0],
            'conversation_id': ', '.join(map(str, x['conversation_id'].unique())),
            'speaker_id': ', '.join(map(str, x['speaker_id'].unique()))  # Assuming each speaker has a unique ID
            }).reset_index()

        conversation_info[["collection_title", "conversation_id", 'speaker_id']] = pd.DataFrame(
        conversation_info[0].tolist(), index=conversation_info.index)
        conversation_info = conversation_info.drop(columns=[0])
        
        # Merge aggregated embeddings with speaker and conversation info
        #self.speaker_embeddings = pd.merge(self.speaker_embeddings, speaker_info, on='speaker_name')
        self.speaker_embeddings = pd.merge(self.speaker_embeddings, conversation_info, on=['collection_id', 'speaker_name'])
        
        # Ensure there are no duplicates for the same speaker name in the same collection
        self.speaker_embeddings.drop_duplicates(subset=['collection_id', 'speaker_name'], inplace=True)


    def compute_umap(self, data):
        scaled_X = StandardScaler().fit_transform(np.vstack(data['Latent-Attention_Embedding'].values))
        reducer = umap.UMAP(n_components=3, random_state=42)
        embedding_2d = reducer.fit_transform(scaled_X)
        return embedding_2d

    def plot_aggregated(self):
        self.compute_aggregated_embeddings()
        embedding_2d = self.compute_umap(self.speaker_embeddings)

        self.speaker_embeddings['UMAP_1'] = embedding_2d[:, 0]
        self.speaker_embeddings['UMAP_2'] = embedding_2d[:, 1]
        self.speaker_embeddings['UMAP_3'] = embedding_2d[:, 2]
        #self.speaker_embeddings['Facilitator'] = self.speaker_embeddings['is_fac'].map(
        #    {True: 'Facilitator', False: 'Participant'})

        custom_color_palette = [
        '#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080', 
        '#FFFF00', '#00FFFF', '#FF00FF', '#808080', '#000000',
        '#FFC0CB', '#800000', '#808000', '#008000', '#000080',
        '#FF4500', '#2E8B57', '#4682B4', '#D2691E', '#9ACD32'
        ]

        fig = px.scatter_3d(   
            self.speaker_embeddings,
            x='UMAP_1', 
            y='UMAP_2',
            z='UMAP_3',
            color='collection_title',  # Use collection_id for color distinction
            title=f'{self.model}: Aggregated Speaker Turn Embeddings for {self.collection_name}',
            hover_name='speaker_name',
            hover_data={
                        "collection_title": True,
                        "conversation_id": True,
                        "speaker_id": True,
                        'UMAP_1': False,
                        'UMAP_2': False,
                        'UMAP_3': False},
            color_discrete_sequence= custom_color_palette # Use a predefined discrete color sequence
        )

        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='black')))
        fig.write_html(self.output_path_template)
        fig.show()
        print(f"Saved aggregated UMAP plot for {self.collection_name}")

model = "nv-embed"
collection_name = "corpus_data"
input_path_template = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\output\umap\data_nv-embed_processed_output.pkl"
output_path_template = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\output\graphs\nv-embed\corpus\global_umap.html"

visualizer = GlobalEmbeddingVisualizer(model, collection_name, input_path_template, output_path_template)
visualizer.plot_aggregated()
