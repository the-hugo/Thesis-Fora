import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

# Load your dataset
df = pd.read_pickle(r'C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\embeddings\data_nv-embed_processed_output.pkl')
df = df[df['annotated']].copy()

# Normalize time within each conversation
df['conversation_duration'] = df.groupby('conversation_id')['audio_end_offset'].transform('max') - \
                                df.groupby('conversation_id')['audio_start_offset'].transform('min')
df['normalized_time'] = 100 * (df['audio_start_offset'] - df.groupby('conversation_id')['audio_start_offset'].transform('min')) / df['conversation_duration']

# Define the facilitation strategy columns
facilitation_columns = [
    'Express affirmation', 'Specific invitation', 'Provide example',
    'Open invitation', 'Make connections', 'Express appreciation', 'Follow up question'
]

# Create time bins (0-100 split into 100 bins)
time_bins = np.linspace(0, 100, 101)

# Initialize an empty list to collect traces
traces = []

# For each facilitation strategy, aggregate over all conversations
for strategy in facilitation_columns:
    # Sum the counts for the current strategy across all conversations per time bin
    strategy_series = df.groupby(pd.cut(df['normalized_time'], bins=time_bins))[strategy].sum()
    # Smooth the time series with a rolling average window
    strategy_series_smooth = strategy_series.rolling(window=5, min_periods=1).mean()
    # Create a trace for the strategy
    traces.append(go.Scatter(
        x=time_bins[:-1],
        y=strategy_series_smooth,
        mode='lines',
        name=strategy,
        line=dict(),
        fill='tozeroy',
        fillcolor='rgba(0, 0, 0, 0)'  # No fill for individual strategies
    ))

# Aggregate an overall facilitation strategy trace (using the max value per row)
df['facilitation_strategy'] = df[facilitation_columns].max(axis=1)
overall_series = df.groupby(pd.cut(df['normalized_time'], bins=time_bins))['facilitation_strategy'].sum()
overall_series_smooth = overall_series.rolling(window=5, min_periods=1).mean()

overall_trace = go.Scatter(
    x=time_bins[:-1],
    y=overall_series_smooth,
    mode='lines',
    name='Overall Facilitation Strategy',
    line=dict(color='black', dash='dash'),
    fill='tozeroy',
    fillcolor='rgba(173, 216, 230, 0.3)'  # light blue with transparency
)
traces.append(overall_trace)

# Aggregate personal sharing (using max of 'Personal story' and 'Personal experience')
df['personal_sharing'] = df[['Personal story', 'Personal experience']].max(axis=1)
personal_series = df.groupby(pd.cut(df['normalized_time'], bins=time_bins))['personal_sharing'].sum()
personal_series_smooth = personal_series.rolling(window=5, min_periods=1).mean()

personal_trace = go.Scatter(
    x=time_bins[:-1],
    y=personal_series_smooth,
    mode='lines',
    name='Personal Sharing',
    line=dict(color='green'),
    fill='tozeroy',
    fillcolor='rgba(144, 238, 144, 0.3)'  # light green with transparency
)
traces.append(personal_trace)

# Define the layout for the graph
layout = go.Layout(
    title={
        'text': 'Aggregated Facilitation Strategies and Personal Sharing (0-100)',
        'font': {'size': 34}
    },
    xaxis=dict(
        title={
            'text': 'Normalized Time (0-100)',
            'font': {'size': 28}
        },
        tickfont={'size': 24}
    ),
    yaxis=dict(
        title={
            'text': 'Count',
            'font': {'size': 28}
        },
        tickfont={'size': 24}
    ),
    legend=dict(
        x=0.05, y=0.95,
        font={'size': 24}
    ),
    width=1920,
    height=1080
)

# Create the figure with all traces and display the aggregated graph
fig = go.Figure(data=traces, layout=layout)
pio.show(fig)
