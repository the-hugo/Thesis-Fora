import umap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px

model = "nv-embed"
collection_name = "collection-150_Maine"
input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\output\{}_{}_processed_output.pkl".format(collection_name, model)
print(f"Loading data from {input_path}")
df = pd.read_pickle(input_path)

df['Latent-Attention_Embedding'] = df['Latent-Attention_Embedding'].apply(np.array)

df = df.dropna(subset=['Latent-Attention_Embedding'])

speaker_embeddings = df.groupby('Speaker Name')['Latent-Attention_Embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()

speaker_info = df[['Speaker Name', 'Is Facilitator']].drop_duplicates()
speaker_embeddings = pd.merge(speaker_embeddings, speaker_info, on='Speaker Name')

X = np.vstack(speaker_embeddings['Latent-Attention_Embedding'].values)
scaled_X = StandardScaler().fit_transform(X)

reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(scaled_X)

speaker_embeddings['UMAP_1'] = embedding_2d[:, 0]
speaker_embeddings['UMAP_2'] = embedding_2d[:, 1]

speaker_embeddings['Facilitator'] = speaker_embeddings['Is Facilitator'].map({True: 'Facilitator', False: 'Non-Facilitator'})

fig = px.scatter(
    speaker_embeddings,
    x='UMAP_1', 
    y='UMAP_2',
    color='Facilitator',
    title='{}: Aggregated Speaker Turn Embeddings for {}'.format(model, collection_name),
    labels={'UMAP_1': 'UMAP Dimension 1', 'UMAP_2': 'UMAP Dimension 2'},
    hover_name='Speaker Name',
    color_discrete_map={'Facilitator': 'red', 'Non-Facilitator': 'blue'}
)

fig.update_traces(marker=dict(size=10, line=dict(width=2, color='black')), selector=dict(name='Facilitator'))

#fig.show()
fig.write_html(f'C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/data/output/{model}_{collection_name}_collection_aggregated_umap.html')
