import umap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import textwrap
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objs as go

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
        
        speaker_info = self.df[['Is Facilitator', "Speaker Name"]].drop_duplicates()

        conversation_info = self.df.groupby('Speaker Name').apply(
            lambda x: {
                'Index in Conversation': ', '.join(map(str, x['Index in Conversation'].values)),
                'Snippet ID': ', '.join(map(str, x['Snippet ID'].values)),
                'Speaker ID': ', '.join(map(str, x['Speaker ID'].unique()))  # Assuming each speaker has a unique ID
            }).reset_index()

        self.speaker_embeddings['Unique Speaker Turns'] = self.df.groupby('Speaker Name')['Snippet ID'].nunique().values
        
        self.speaker_embeddings['Average Turn Length'] = self.df.groupby('Speaker Name')['duration'].mean().values

        self.df['Normalized Turn Position'] = self.df.groupby('Conversation ID')['Index in Conversation'].apply(lambda x: x / x.max())

        self.speaker_embeddings['Average Turn Position'] = self.df.groupby('Speaker Name')['Normalized Turn Position'].mean().values

        conversation_info[['Index in Conversation', 'Snippet ID', 'Speaker ID']] = pd.DataFrame(
            conversation_info[0].tolist(), index=conversation_info.index)
        conversation_info = conversation_info.drop(columns=[0])

        self.speaker_embeddings = pd.merge(self.speaker_embeddings, speaker_info, on='Speaker Name')
        self.speaker_embeddings = pd.merge(self.speaker_embeddings, conversation_info, on='Speaker Name')
        
        facilitator_df = self.speaker_embeddings[self.speaker_embeddings['Is Facilitator'] == True]
        participant_df = self.speaker_embeddings[self.speaker_embeddings['Is Facilitator'] == False]

        facilitator_metrics = facilitator_df[['Unique Speaker Turns', 'Average Turn Length', 'Average Turn Position']].mean()
        participant_metrics = participant_df[['Unique Speaker Turns', 'Average Turn Length', 'Average Turn Position']].mean()

        inter_group_differences = facilitator_metrics - participant_metrics
        
        self.speaker_embeddings['Facilitator_Participant_Differences'] = inter_group_differences

        return self.speaker_embeddings

    def compute_umap(self, data):
        scaled_X = StandardScaler().fit_transform(np.vstack(data['Latent-Attention_Embedding'].values))
        reducer = umap.UMAP(n_components=3, random_state=42)
        embedding_2d = reducer.fit_transform(scaled_X)
        return embedding_2d
    
    def calculate_metrics(self):
        # Calculate intra-metrics (within facilitators and participants)
        facilitator_df = self.speaker_embeddings[self.speaker_embeddings['Is Facilitator'] == True]
        participant_df = self.speaker_embeddings[self.speaker_embeddings['Is Facilitator'] == False]

        # Intra-metrics: Calculate mean values for facilitators and participants separately
        facilitator_metrics = facilitator_df[['Unique Speaker Turns', 'Average Turn Length', 'Average Turn Position']].mean()
        participant_metrics = participant_df[['Unique Speaker Turns', 'Average Turn Length', 'Average Turn Position']].mean()

        # Inter-metrics: Calculate the differences between facilitators and participants
        inter_group_differences = facilitator_metrics - participant_metrics

        # Add metrics to the speaker embeddings dataframe
        self.speaker_embeddings['Intra Turn Length (Facilitator)'] = facilitator_metrics['Average Turn Length']
        self.speaker_embeddings['Intra Turn Length (Participant)'] = participant_metrics['Average Turn Length']
        self.speaker_embeddings['Inter Turn Length Difference'] = inter_group_differences['Average Turn Length']
        
        self.speaker_embeddings['Intra Turn Count (Facilitator)'] = facilitator_metrics['Unique Speaker Turns']
        self.speaker_embeddings['Intra Turn Count (Participant)'] = participant_metrics['Unique Speaker Turns']
        self.speaker_embeddings['Inter Turn Count Difference'] = inter_group_differences['Unique Speaker Turns']
        
        self.speaker_embeddings['Intra Turn Position (Facilitator)'] = facilitator_metrics['Average Turn Position']
        self.speaker_embeddings['Intra Turn Position (Participant)'] = participant_metrics['Average Turn Position']
        self.speaker_embeddings['Inter Turn Position Difference'] = inter_group_differences['Average Turn Position']

        return facilitator_metrics, participant_metrics, inter_group_differences

    def plot_aggregated(self):
        self.compute_aggregated_embeddings()
        
        embedding_3d = self.compute_umap(self.speaker_embeddings)
        self.speaker_embeddings['UMAP_1'] = embedding_3d[:, 0]
        self.speaker_embeddings['UMAP_2'] = embedding_3d[:, 1]
        self.speaker_embeddings['UMAP_3'] = embedding_3d[:, 2]

        self.calculate_metrics()

        self.speaker_embeddings['Index in Conversation'] = self.speaker_embeddings['Index in Conversation'].apply(
            lambda x: '<br>'.join(textwrap.wrap(x, width=50)))
        self.speaker_embeddings['Speaker ID'] = self.speaker_embeddings['Speaker ID'].apply(
            lambda x: '<br>'.join(textwrap.wrap(x, width=50)))

        self.speaker_embeddings['Role'] = self.speaker_embeddings['Is Facilitator'].map({True: 'Facilitator', False: 'Participant'})

        fig = px.scatter_3d(
            self.speaker_embeddings,
            x='UMAP_1', 
            y='UMAP_2',
            z='UMAP_3',  # Third dimension for 3D visualization
            color='Role',  # Use the 'Role' column to color by role (Facilitator/Participant)
            title=f'{self.model}: Aggregated Speaker Turn Embeddings for {self.collection_name}',
            hover_name='Speaker Name',
            hover_data={
                'Index in Conversation': True,
                "Average Turn Length": True,       
                "Unique Speaker Turns": True,        
                "Average Turn Position": True,       
                "Speaker ID": True,
                'Intra Turn Length (Facilitator)': True,   
                'Intra Turn Length (Participant)': True,
                'Inter Turn Length Difference': True,      
                'Intra Turn Count (Facilitator)': True,    
                'Intra Turn Count (Participant)': True,    
                'Inter Turn Count Difference': True,       
                'Intra Turn Position (Facilitator)': True, 
                'Intra Turn Position (Participant)': True, 
                'Inter Turn Position Difference': True,    
                'UMAP_1': False,            
                'UMAP_2': False,
                'UMAP_3': False
            },
            color_discrete_map={'Facilitator': 'red', 'Participant': 'blue'}
        )

        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='black')), selector=dict(name='Facilitator'))

        fig.write_html(self.output_path_template.format(self.model, self.collection_name, 'collection_aggregated_umap_3d'))
        print(f"Saved 3D aggregated UMAP plot for {self.collection_name}")

    def compute_auto_correlation(self, group, max_lag=15):
        """
        Compute auto-correlation for contributions at different time lags, grouped by speaker.
        """
        auto_corr_matrix = []

        grouped = group.groupby('Speaker Name')

        for speaker, speaker_group in grouped:
            embeddings = np.vstack(speaker_group['Latent-Attention_Embedding'].values)
            speaker_auto_corr_matrix = np.zeros((len(embeddings), max_lag))
            
            for lag in range(1, max_lag + 1):
                for i in range(len(embeddings) - lag):
                    speaker_auto_corr_matrix[i, lag-1] = cosine_similarity(
                        embeddings[i].reshape(1, -1),
                        embeddings[i + lag].reshape(1, -1)
                    )[0][0]
                
                auto_corr_matrix.append(speaker_auto_corr_matrix)

        return np.vstack(auto_corr_matrix)

    def compute_cross_correlation(self, group, max_lag=15):
        """
        Compute cross-correlation for contributions made by different speakers at different time lags.
        Cross-correlations are only computed between different speakers.
        """
        embeddings = np.vstack(group['Latent-Attention_Embedding'].values)
        speaker_ids = group['Speaker ID'].values
        cross_corr_matrix = np.zeros((len(embeddings), len(embeddings), max_lag))

        for lag in range(1, max_lag + 1):
            for i in range(len(embeddings) - lag):
                for j in range(i + lag, len(embeddings)):
                    if speaker_ids[i] != speaker_ids[j]:
                        cross_corr_matrix[i, j, lag-1] = cosine_similarity(
                            embeddings[i].reshape(1, -1),
                            embeddings[j].reshape(1, -1)
                        )[0][0]

        return cross_corr_matrix

    def plot_cross_auto_correlation_3d(self, group, auto_corr_matrix, cross_corr_matrix, threshold=0.6):
        """
        Plot 3D UMAP embeddings and connect contributions with significant cross- or auto-correlation.
        """

        embedding_3d = self.compute_umap(group)
        group['UMAP_1'] = embedding_3d[:, 0]
        group['UMAP_2'] = embedding_3d[:, 1]
        group['UMAP_3'] = embedding_3d[:, 2]
        group['Facilitator'] = group['Is Facilitator'].map({True: 'Facilitator', False: 'Non-Facilitator'})

        group['Wrapped Content'] = group['Content'].apply(lambda x: '<br>'.join(textwrap.wrap(x, width=50)))
        group['duration'] = group['duration'].astype(int)

        fig = go.Figure()

        for role, color in zip(['Facilitator', 'Non-Facilitator'], ['red', 'blue']):
            role_group = group[group['Facilitator'] == role]
            fig.add_trace(go.Scatter3d(
                x=role_group['UMAP_1'],
                y=role_group['UMAP_2'],
                z=role_group['UMAP_3'],
                mode='markers',
                marker=dict(color=color, size=8, line=dict(width=2, color='black' if role == 'Facilitator' else 'white')),
                name=role,
                customdata=np.stack((
                    role_group['Speaker Name'],
                    role_group['Index in Conversation'],
                    role_group['Wrapped Content'],
                    role_group['duration']
                ), axis=-1),
                hovertemplate="""
                    Speaker Name: %{customdata[0]}<br>
                    Index in Conversation: %{customdata[1]}<br>
                    Content: %{customdata[2]}<br>
                    Duration: %{customdata[3]} seconds
                    <extra></extra>
                """
            ))

        first_auto_corr = True
        first_cross_corr = True

        num_contributions = len(group)
        for i in range(num_contributions):
            current_speaker = group.iloc[i]['Speaker ID']
            
            for lag in range(auto_corr_matrix.shape[1]):
                if auto_corr_matrix[i, lag] > threshold and i + lag < num_contributions:
                    if group.iloc[i + lag]['Speaker ID'] == current_speaker:
                        # Auto-correlation: Same speaker
                        fig.add_trace(go.Scatter3d(
                            x=[group.iloc[i]['UMAP_1'], group.iloc[i + lag]['UMAP_1']],
                            y=[group.iloc[i]['UMAP_2'], group.iloc[i + lag]['UMAP_2']],
                            z=[group.iloc[i]['UMAP_3'], group.iloc[i + lag]['UMAP_3']],
                            mode='lines',
                            line=dict(color='orange', width=2),
                            hoverinfo='skip',
                            showlegend=first_auto_corr,
                            name='Auto-correlation'
                        ))
                        first_auto_corr = False

            for j in range(i + 1, num_contributions):
                for lag in range(cross_corr_matrix.shape[2]):
                    if cross_corr_matrix[i, j, lag] > threshold:
                        fig.add_trace(go.Scatter3d(
                            x=[group.iloc[i]['UMAP_1'], group.iloc[j]['UMAP_1']],
                            y=[group.iloc[i]['UMAP_2'], group.iloc[j]['UMAP_2']],
                            z=[group.iloc[i]['UMAP_3'], group.iloc[j]['UMAP_3']],
                            mode='lines',
                            line=dict(color='green', width=1),
                            hoverinfo='skip',
                            showlegend=first_cross_corr,
                            name='Cross-correlation'
                        ))
                        first_cross_corr = False

        fig.update_layout(
            title=f'Cross- and Auto-Correlation (3D) for Conversation {group["Conversation ID"].iloc[0]}',
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3'
            ),
            showlegend=True
        )

        conversation_id = group['Conversation ID'].iloc[0]
        fig.write_html(self.output_path_template.format(self.model, self.collection_name, f'conversation_{conversation_id}_cross_auto_correlation_3d'))

    def plot_by_conversation_with_cross_auto_correlation_3d(self, max_lag=5, threshold=0.6):
        """
        Plot 3D UMAP for each conversation and connect turns with significant cross- and auto-correlation.
        """
        for conversation_id, group in self.df.groupby('Conversation ID'):
            auto_corr_matrix = self.compute_auto_correlation(group, max_lag=max_lag)
            cross_corr_matrix = self.compute_cross_correlation(group, max_lag=max_lag)

            self.plot_cross_auto_correlation_3d(group, auto_corr_matrix, cross_corr_matrix, threshold)

model = "nv-embed"
collection_name = "collection-150_Maine"
input_path_template = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\output\{}_{}_processed_output.pkl"
output_path_template = r'C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/data/output/{}_{}_{}.html'

visualizer = EmbeddingVisualizer(model, collection_name, input_path_template, output_path_template)
visualizer.plot_aggregated()
visualizer.plot_by_conversation_with_cross_auto_correlation_3d()
