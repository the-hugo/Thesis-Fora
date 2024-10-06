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

        grouped = facilitators_df.groupby('collection_id')

        similarity_matrices = {}

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
group_columns = ['collection_id', 'speaker_name', 'is_fac']
similarity_calculator = FacilitatorSimilarity(input_path, group_columns)
similarity_calculator.load_data()

similarity_calculator.compute_average_embeddings()

similarity_matrices = similarity_calculator.compute_similarity_matrices()

results = similarity_calculator.run_permutation_tests()
#similarity_calculator.plot_similarity_matrices()
