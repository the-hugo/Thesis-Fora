import umap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Load the dataset
collection_name = "collection-24_UnitedWayDane"
input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\output\{}_nv-embed_processed_output.pkl".format(collection_name)
print(f"Loading data from {input_path}")
df = pd.read_pickle(input_path)

# Clean and scale data: Ensure 'Latent-Attention_Embedding' is in proper array format
df['Latent-Attention_Embedding'] = df['Latent-Attention_Embedding'].apply(np.array)

# Handle any missing values (optional, depending on your dataset)
df = df.dropna(subset=['Latent-Attention_Embedding'])

# Stack the embeddings into a 2D numpy array
X = np.vstack(df['Latent-Attention_Embedding'].values)

# Standardize the embeddings for comparability
scaled_X = StandardScaler().fit_transform(X)

# Step 2: Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(scaled_X)

# Step 3: Add UMAP embeddings to the dataframe for easy plotting
df['UMAP_1'] = embedding_2d[:, 0]
df['UMAP_2'] = embedding_2d[:, 1]

# Map facilitator and non-facilitator to colors
df['Facilitator'] = df['Is Facilitator'].map({True: 'Facilitator', False: 'Non-Facilitator'})

# Plot the UMAP embedding using Plotly Express
fig = px.scatter(
    df,
    x='UMAP_1', 
    y='UMAP_2',
    color='Facilitator',  # Color by facilitator status
    title='UMAP Visualization of Latent-Attention Embedding by Speaker',
    labels={'UMAP_1': 'UMAP Dimension 1', 'UMAP_2': 'UMAP Dimension 2'},
    hover_name='Speaker Name',  # Hover will display speaker names
    color_discrete_map={'Facilitator': 'red', 'Non-Facilitator': 'blue'}  # Set color map
)

# Customize marker size for facilitators
fig.update_traces(marker=dict(size=10, line=dict(width=2, color='black')), selector=dict(name='Facilitator'))

# Show the plot
fig.show()
