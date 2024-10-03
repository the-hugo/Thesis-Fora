import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

# Load your dataset
df = pd.read_pickle(r'C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\output\umap\data_nv-embed_processed_output.pkl')
df = df[df['annotated']].copy()

# Normalize time within each conversation
df['conversation_duration'] = df.groupby('conversation_id')['audio_end_offset'].transform('max') - df.groupby('conversation_id')['audio_start_offset'].transform('min')
df['normalized_time'] = 100 * (df['audio_start_offset'] - df.groupby('conversation_id')['audio_start_offset'].transform('min')) / df['conversation_duration']

# Define the facilitation strategy columns
facilitation_columns = [
    'Express affirmation', 'Specific invitation', 'Provide example',
    'Open invitation', 'Make connections', 'Express appreciation', 'Follow up question'
]

# Time bins for normalized time
time_bins = np.linspace(0, 100, 101)

# Get unique collections
collections = df['collection_id'].unique()

# Create an empty list to store traces for each collection
traces = []

# Loop through each collection and calculate the normalized overall facilitation strategy time series
for collection in collections:
    df_collection = df[df['collection_id'] == collection]
    
    # Add an overall strategy trace (sum of all strategies) for this collection
    df_collection['facilitation_strategy'] = df_collection[facilitation_columns].max(axis=1)
    overall_facilitation_time_series = df_collection.groupby(pd.cut(df_collection['normalized_time'], bins=time_bins))['facilitation_strategy'].sum()
    
    # Normalize the counts by the total count of facilitation strategies in this collection
    total_facilitation_count = df_collection['facilitation_strategy'].sum()
    overall_facilitation_time_series_normalized = overall_facilitation_time_series / total_facilitation_count
    
    # Smooth the normalized series
    overall_facilitation_time_series_normalized_smooth = overall_facilitation_time_series_normalized.rolling(window=5, min_periods=1).mean()

    traces.append(go.Scatter(
        x=time_bins[:-1],
        y=overall_facilitation_time_series_normalized_smooth,
        mode='lines',
        name=f'Overall Facilitation Strategy (Collection {collection})',
        line=dict(dash='solid'),
        fill='tozeroy',
        fillcolor='rgba(173, 216, 230, 0.3)'  # lightblue with transparency for the overall line
    ))

# Set up the layout
layout = go.Layout(
    title='Normalized Overall Facilitation Strategy Use by Collection over Time (0-100)',
    xaxis=dict(title='Normalized Time (0-100)'),
    yaxis=dict(title='Proportion of Total Facilitation Strategies'),
    legend=dict(x=0.05, y=0.95)
)

# Create the figure
fig = go.Figure(data=traces, layout=layout)

# Show the plot
pio.show(fig)

# Save the plot as an HTML file
pio.write_html(fig, file='normalized_overall_facilitation_strategy_by_collection.html', auto_open=True)
