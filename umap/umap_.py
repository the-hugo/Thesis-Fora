import umap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import textwrap

class EmbeddingVisualizer:
    def __init__(self, model, collection_name, input_path_template, output_path_template):
        self.model = model
        self.collection_name = collection_name
        self.input_path = input_path_template.format(collection_name, model)
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

        self.speaker_embeddings = self.df.groupby('Speaker Name')['Latent-Attention_Embedding'].apply(
            lambda x: np.mean(np.vstack(x), axis=0)).reset_index()
        
        # Get unique speaker information (is facilitator or not)
        speaker_info = self.df[['Is Facilitator', "Speaker Name"]].drop_duplicates()

        # Compute conversation-related information for each speaker
        conversation_info = self.df.groupby('Speaker Name').apply(
            lambda x: {
                'Index in Conversation': ', '.join(map(str, x['Index in Conversation'].values)),
                'Snippet ID': ', '.join(map(str, x['Snippet ID'].values)),
                'Speaker ID': ', '.join(map(str, x['Speaker ID'].unique()))  # Assuming each speaker has a unique ID
            }).reset_index()

        # Add the number of unique speaker turns (turn count)
        self.speaker_embeddings['Unique Speaker Turns'] = self.df.groupby('Speaker Name')['Snippet ID'].nunique().values
        
        # Add the average turn length (mean of the 'Duration' column for each speaker)
        self.speaker_embeddings['Average Turn Length'] = self.df.groupby('Speaker Name')['duration'].mean().values

        # Normalize 'Index in Conversation' to calculate relative turn position (0 = beginning, 1 = end)
        self.df['Normalized Turn Position'] = self.df.groupby('Conversation ID')['Index in Conversation'].apply(lambda x: x / x.max())

        # Calculate the average turn position for each speaker (mean of normalized turn positions)
        self.speaker_embeddings['Average Turn Position'] = self.df.groupby('Speaker Name')['Normalized Turn Position'].mean().values

        # Convert the conversation-related information into separate columns
        conversation_info[['Index in Conversation', 'Snippet ID', 'Speaker ID']] = pd.DataFrame(
            conversation_info[0].tolist(), index=conversation_info.index)
        conversation_info = conversation_info.drop(columns=[0])

        # Merge speaker embeddings with speaker info (facilitator or not) and conversation info
        self.speaker_embeddings = pd.merge(self.speaker_embeddings, speaker_info, on='Speaker Name')
        self.speaker_embeddings = pd.merge(self.speaker_embeddings, conversation_info, on='Speaker Name')
        
        # Add intra and inter group comparison for facilitators and participants
        facilitator_df = self.speaker_embeddings[self.speaker_embeddings['Is Facilitator'] == True]
        participant_df = self.speaker_embeddings[self.speaker_embeddings['Is Facilitator'] == False]

        # Calculate intra-group averages (facilitators and participants separately)
        facilitator_metrics = facilitator_df[['Unique Speaker Turns', 'Average Turn Length', 'Average Turn Position']].mean()
        participant_metrics = participant_df[['Unique Speaker Turns', 'Average Turn Length', 'Average Turn Position']].mean()

        # Calculate differences (inter-group comparison)
        inter_group_differences = facilitator_metrics - participant_metrics
        
        # Store inter-group comparison results
        self.speaker_embeddings['Facilitator_Participant_Differences'] = inter_group_differences

        return self.speaker_embeddings

    def compute_umap(self, data):
        scaled_X = StandardScaler().fit_transform(np.vstack(data['Latent-Attention_Embedding'].values))
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(scaled_X)
        return embedding_2d

    def plot_aggregated(self):
            # Compute the aggregated embeddings
        self.compute_aggregated_embeddings()
        
        # Compute UMAP embeddings for visualization
        embedding_2d = self.compute_umap(self.speaker_embeddings)
        self.speaker_embeddings['UMAP_1'] = embedding_2d[:, 0]
        self.speaker_embeddings['UMAP_2'] = embedding_2d[:, 1]

        # Map facilitators and participants for coloring
        self.speaker_embeddings['Facilitator'] = self.speaker_embeddings['Is Facilitator'].map(
            {True: 'Facilitator', False: 'Participant'})

        # Calculate intra-metrics (within facilitators and participants)
        facilitator_df = self.speaker_embeddings[self.speaker_embeddings['Is Facilitator'] == True]
        participant_df = self.speaker_embeddings[self.speaker_embeddings['Is Facilitator'] == False]

        # Intra-metrics: Calculate mean values for facilitators and participants separately
        facilitator_metrics = facilitator_df[['Unique Speaker Turns', 'Average Turn Length', 'Average Turn Position']].mean()
        participant_metrics = participant_df[['Unique Speaker Turns', 'Average Turn Length', 'Average Turn Position']].mean()

        # Inter-metrics: Calculate the differences between facilitators and participants
        inter_group_differences = facilitator_metrics - participant_metrics

        # Add the inter and intra metrics into hover information
        self.speaker_embeddings['Intra Turn Length (Facilitator)'] = facilitator_metrics['Average Turn Length']
        self.speaker_embeddings['Intra Turn Length (Participant)'] = participant_metrics['Average Turn Length']
        self.speaker_embeddings['Inter Turn Length Difference'] = inter_group_differences['Average Turn Length']
        
        self.speaker_embeddings['Intra Turn Count (Facilitator)'] = facilitator_metrics['Unique Speaker Turns']
        self.speaker_embeddings['Intra Turn Count (Participant)'] = participant_metrics['Unique Speaker Turns']
        self.speaker_embeddings['Inter Turn Count Difference'] = inter_group_differences['Unique Speaker Turns']
        
        self.speaker_embeddings['Intra Turn Position (Facilitator)'] = facilitator_metrics['Average Turn Position']
        self.speaker_embeddings['Intra Turn Position (Participant)'] = participant_metrics['Average Turn Position']
        self.speaker_embeddings['Inter Turn Position Difference'] = inter_group_differences['Average Turn Position']

        # Wrap long text in 'Index in Conversation' and 'Speaker ID' columns for better visualization
        self.speaker_embeddings['Index in Conversation'] = self.speaker_embeddings['Index in Conversation'].apply(
            lambda x: '<br>'.join(textwrap.wrap(x, width=50)))
        self.speaker_embeddings['Speaker ID'] = self.speaker_embeddings['Speaker ID'].apply(
            lambda x: '<br>'.join(textwrap.wrap(x, width=50)))

        # Create the scatter plot using UMAP dimensions with updated hover information
        fig = px.scatter(
            self.speaker_embeddings,
            x='UMAP_1', 
            y='UMAP_2',
            color='Facilitator',  # Color by role (Facilitator/Participant)
            title=f'{self.model}: Aggregated Speaker Turn Embeddings for {self.collection_name}',
            hover_name='Speaker Name',
            hover_data={
                'Index in Conversation': True,
                "Average Turn Length": True,         # Show average turn length in hover
                "Unique Speaker Turns": True,        # Show unique speaker turns in hover
                "Average Turn Position": True,       # Show average turn position in hover
                "Speaker ID": True,
                'Intra Turn Length (Facilitator)': True,   # Intra metric for turn length (Facilitator)
                'Intra Turn Length (Participant)': True,   # Intra metric for turn length (Participant)
                'Inter Turn Length Difference': True,      # Inter metric for turn length
                'Intra Turn Count (Facilitator)': True,    # Intra metric for turn count (Facilitator)
                'Intra Turn Count (Participant)': True,    # Intra metric for turn count (Participant)
                'Inter Turn Count Difference': True,       # Inter metric for turn count
                'Intra Turn Position (Facilitator)': True, # Intra metric for turn position (Facilitator)
                'Intra Turn Position (Participant)': True, # Intra metric for turn position (Participant)
                'Inter Turn Position Difference': True,    # Inter metric for turn position
                'Facilitator': False,                      # Facilitate visualization by omitting redundant info
                'UMAP_1': False,            
                'UMAP_2': False
            },
            color_discrete_map={'Facilitator': 'red', 'Participant': 'blue'}
        )

        # Update trace appearance (marker size, border for facilitators)
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='black')), selector=dict(name='Facilitator'))

        # Save the plot to HTML file
        fig.write_html(self.output_path_template.format(self.model, self.collection_name, 'collection_aggregated_umap'))
        print(f"Saved aggregated UMAP plot for {self.collection_name}")

    def plot_by_conversation(self):
        for conversation_id, group in self.df.groupby('Conversation ID'):
            embedding_2d = self.compute_umap(group)
            group['UMAP_1'] = embedding_2d[:, 0]
            group['UMAP_2'] = embedding_2d[:, 1]
            group['Facilitator'] = group['Is Facilitator'].map({True: 'Facilitator', False: 'Non-Facilitator'})
            group['Wrapped Content'] = group['Content'].apply(lambda x: '<br>'.join(textwrap.wrap(x, width=50)))
            group['duration'] = group['duration'].astype(int)

            fig = px.scatter(
                group,
                x='UMAP_1', 
                y='UMAP_2',
                color='Facilitator',
                title=f'{self.model}: Embedding for Conversation {conversation_id} in {self.collection_name}',
                #labels={'Words': 'Content', "Duration": "duration"},
                hover_name='Speaker Name',
                hover_data={'Index in Conversation': True, "Wrapped Content": True, "duration": True,
                            'Facilitator': False,
                            'UMAP_1': False,            
                            'UMAP_2': False  },
                color_discrete_map={'Facilitator': 'red', 'Non-Facilitator': 'blue'}
            )
            fig.update_traces(marker=dict(size=10, line=dict(width=2, color='black')), selector=dict(name='Facilitator'))
            fig.write_html(self.output_path_template.format(self.model, self.collection_name, f'conversation_{conversation_id}_umap'))
            print(f"Saved UMAP plot for Conversation {conversation_id} in {self.collection_name}")


# Usage example
model = "nv-embed"
collection_name = "collection-150_Maine"
input_path_template = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\output\{}_{}_processed_output.pkl"
output_path_template = r'C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/data/output/{}_{}_{}.html'

visualizer = EmbeddingVisualizer(model, collection_name, input_path_template, output_path_template)
visualizer.plot_aggregated()
visualizer.plot_by_conversation()
