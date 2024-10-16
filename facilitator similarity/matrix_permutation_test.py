import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff
import random
from sklearn.metrics.pairwise import cosine_similarity

class FacilitatorSimilarity:
    def __init__(self, input_path, group_columns):
        self.input_path = input_path
        self.df = None
        self.group_columns = group_columns
        self.similarity_matrices = None

    def load_data(self):
        print(f"Loading data from {self.input_path}")
        self.df = pd.read_pickle(self.input_path)
        self.df['speaker_name'] = self.df['speaker_name'].str.lower().str.strip()
        self.df = self.df[~self.df['speaker_name'].str.contains('interpreter|other speaker|computer voice|audio|highlight|interviewer|multiple voices|group|facilitator|video|participant|^speaker')]

    def compute_average_embeddings(self):
        self.df = self.df.dropna(subset=['Latent-Attention_Embedding'])
        self.speaker_embeddings = self.df.groupby(self.group_columns).agg(
            Latent_Attention_Embedding=('Latent-Attention_Embedding', lambda x: np.mean(np.stack(x), axis=0)),
        ).reset_index()

    def compute_similarity_matrices(self):
        facilitators_df = self.speaker_embeddings[self.speaker_embeddings['is_fac'] == True]

        # Step 1: Calculate the total number of unique conversations per collection
        total_conversations = facilitators_df.groupby('collection_id')['conversation_id'].nunique()

        # Step 2: Calculate the number of conversations per facilitator (grouped by facilitator_name)
        facilitator_conversations = facilitators_df.groupby(['collection_id', 'speaker_name'])['conversation_id'].count()

        # Step 3: Calculate the ratio of conversations per facilitator as a percentage
        ratio = (100 * (facilitator_conversations / total_conversations)).round(2)

        # Step 4: Merge facilitators with the same speaker name (up to the first space) within a collection
        # Extract the first name (up to the first space) for grouping purposes
        facilitators_df['merged_name'] = facilitators_df['speaker_name'].str.split().str[0]

        # Step 5: Group by 'collection_id' and 'merged_name', then aggregate conversation count and ratio
        merged_conversations = facilitators_df.groupby(['collection_id', 'merged_name'])['conversation_id'].count()

        # Recalculate the total conversations per merged name
        merged_total_conversations = facilitators_df.groupby('collection_id')['conversation_id'].nunique()

        # Recalculate the ratio of conversations per merged facilitator
        merged_ratio = (100 * (merged_conversations / merged_total_conversations)).round(2)

        # Step 6: Reset index to prepare for merging with the original dataframe
        facilitator_conversations = facilitator_conversations.reset_index(name='conversation_count')
        ratio = ratio.reset_index(name='conversation_ratio')

        # Step 7: Merge the facilitator conversation count and ratio back into the original dataframe
        facilitators_df = facilitators_df.merge(facilitator_conversations, on=['collection_id', 'speaker_name'], how='left')
        facilitators_df = facilitators_df.merge(ratio, on=['collection_id', 'speaker_name'], how='left')

        # Step 8: Only include facilitators that have been in at least 2 conversations
        facilitators_df = facilitators_df[facilitators_df['conversation_count'] >= 2]
        
        grouped = facilitators_df.groupby('collection_id')

        similarity_matrices = {}
        
        # print the facilitator name, amount of conversations they have been in and the ratio of conversations they have been in and the collection_id
        facilitator_summary = facilitators_df[['speaker_name', 'conversation_count', 'conversation_ratio', 'collection_id']].drop_duplicates()

        # Print the facilitator summary
        for index, row in facilitator_summary.iterrows():
            print(f"Facilitator: {row['speaker_name']}, Collection ID: {row['collection_id']}, "
                f"Conversations: {row['conversation_count']}, Ratio: {row['conversation_ratio']}%")

        for collection_id, group in grouped:
            facilitator_names = group['speaker_name'].values
            embeddings = np.stack(group['Latent_Attention_Embedding'].values)

            similarity_matrix = cosine_similarity(embeddings)
            
            similarity_matrices[collection_id] = (facilitator_names, similarity_matrix)
        
        self.similarity_matrices = similarity_matrices
        return similarity_matrices

    def permutation_test(self, matrix, num_permutations=1000):
        diagonal_values = np.diag(matrix)
        off_diagonal_values = matrix[np.triu_indices_from(matrix, k=1)]

        actual_diff = np.mean(diagonal_values) - np.mean(off_diagonal_values)

        perm_diffs = []
        combined_values = np.concatenate([diagonal_values, off_diagonal_values])

        for _ in range(num_permutations):
            random.shuffle(combined_values)
            perm_self = combined_values[:len(diagonal_values)]
            perm_others = combined_values[len(diagonal_values):]
            perm_diffs.append(np.mean(perm_self) - np.mean(perm_others))

        p_value = np.mean(np.array(perm_diffs) >= actual_diff)

        return actual_diff, p_value

    def run_permutation_tests(self, num_permutations=1000):
        results = []
        for collection_id, (names, matrix) in self.similarity_matrices.items():
            actual_diff, p_value = self.permutation_test(matrix, num_permutations)
            results.append({"Collection": collection_id, "Actual Difference": actual_diff, "P-Value": p_value})
            print(f"Collection {collection_id}: Actual Difference = {actual_diff:.6f}, P-Value = {p_value:.4f}")
        return pd.DataFrame(results)

    def plot_similarity_matrices(self):
        for collection_id, (names, matrix) in self.similarity_matrices.items():
            heatmap = go.Heatmap(
                z=matrix,
                x=names,
                y=names,
                colorscale='Viridis'
            )

            layout = go.Layout(
                title=f'Similarity Matrix for Collection {collection_id}',
                xaxis=dict(title='Facilitators'),
                yaxis=dict(title='Facilitators')
            )

            fig = go.Figure(data=[heatmap], layout=layout)
            fig.show()

input_path = "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/data/output/umap/data_nv-embed_processed_output.pkl"
group_columns = ['collection_id', 'conversation_id', 'speaker_name', 'is_fac']
similarity_calculator = FacilitatorSimilarity(input_path, group_columns)
similarity_calculator.load_data()

similarity_calculator.compute_average_embeddings()

similarity_matrices = similarity_calculator.compute_similarity_matrices()

results = similarity_calculator.run_permutation_tests()
similarity_calculator.plot_similarity_matrices()
