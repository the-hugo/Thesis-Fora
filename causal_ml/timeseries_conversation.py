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

# Aggregate facilitation strategies over normalized time using count for each conversation
time_bins = np.linspace(0, 100, 101)

# Iterate over each conversation and plot its data
for conversation_id, conversation_df in df.groupby('conversation_id'):
    
    # Dictionary to hold smoothed time series for each strategy
    facilitation_time_series_dict = {}

    # Calculate the count for each strategy and smooth the line for the current conversation
    for strategy in facilitation_columns:
        facilitation_time_series = conversation_df.groupby(pd.cut(conversation_df['normalized_time'], bins=time_bins))[strategy].sum()
        facilitation_time_series_smooth = facilitation_time_series.rolling(window=5, min_periods=1).mean()
        facilitation_time_series_dict[strategy] = facilitation_time_series_smooth

    # Create traces for each facilitation strategy
    traces = []
    for strategy, smoothed_series in facilitation_time_series_dict.items():
        traces.append(go.Scatter(
            x=time_bins[:-1],
            y=smoothed_series,
            mode='lines',
            name=strategy,
            line=dict(),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 0, 0)'  # No fill for individual strategies
        ))

    # Add an overall strategy trace (sum of all strategies)
    conversation_df['facilitation_strategy'] = conversation_df[facilitation_columns].max(axis=1)
    overall_facilitation_time_series = conversation_df.groupby(pd.cut(conversation_df['normalized_time'], bins=time_bins))['facilitation_strategy'].sum()
    overall_facilitation_time_series_smooth = overall_facilitation_time_series.rolling(window=5, min_periods=1).mean()

    overall_trace = go.Scatter(
        x=time_bins[:-1],
        y=overall_facilitation_time_series_smooth,
        mode='lines',
        name='Overall Facilitation Strategy',
        line=dict(color='black', dash='dash'),
        fill='tozeroy',
        fillcolor='rgba(173, 216, 230, 0.3)'  # lightblue with transparency for the overall line
    )

    # Add the overall trace to the list of traces
    traces.append(overall_trace)

    # Now, calculate and plot personal sharing for the current conversation
    conversation_df['personal_sharing'] = conversation_df[['Personal story', 'Personal experience']].max(axis=1)
    personal_sharing_time_series = conversation_df.groupby(pd.cut(conversation_df['normalized_time'], bins=time_bins))['personal_sharing'].sum()
    personal_sharing_time_series_smooth = personal_sharing_time_series.rolling(window=5, min_periods=1).mean()

    personal_sharing_trace = go.Scatter(
        x=time_bins[:-1],
        y=personal_sharing_time_series_smooth,
        mode='lines',
        name='Personal Sharing',
        line=dict(color='green'),
        fill='tozeroy',
        fillcolor='rgba(144, 238, 144, 0.3)'  # lightgreen with transparency
    )

    # Add the personal sharing trace to the list of traces
    traces.append(personal_sharing_trace)

    # Set up the layout
    layout = go.Layout(
        title=f'Facilitation Strategies and Personal Sharing for Conversation {conversation_id} (0-100)',
        xaxis=dict(title='Normalized Time (0-100)'),
        yaxis=dict(title='Count'),
        legend=dict(x=0.05, y=0.95)
    )

    # Create the figure for this conversation
    fig = go.Figure(data=traces, layout=layout)

    # Show the plot for the current conversation
    pio.show(fig)
