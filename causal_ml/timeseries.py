import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

df = pd.read_pickle(r'C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\code\data\output\umap\data_nv-embed_processed_output.pkl')
df = df[df['annotated']].copy()

df['conversation_duration'] = df.groupby('conversation_id')['audio_end_offset'].transform('max') - df.groupby('conversation_id')['audio_start_offset'].transform('min')
df['normalized_time'] = 100 * (df['audio_start_offset'] - df.groupby('conversation_id')['audio_start_offset'].transform('min')) / df['conversation_duration']

facilitation_columns = [
    'Express affirmation', 'Specific invitation', 'Provide example',
    'Open invitation', 'Make connections', 'Express appreciation', 'Follow up question'
]

time_bins = np.linspace(0, 100, 101)

traces = []

df['facilitation_strategy'] = df[facilitation_columns].max(axis=1)

overall_facilitation_time_series = df.groupby(pd.cut(df['normalized_time'], bins=time_bins))['facilitation_strategy'].sum()

total_facilitation_count = overall_facilitation_time_series.sum()
overall_facilitation_time_series_normalized = overall_facilitation_time_series / total_facilitation_count

if isinstance(overall_facilitation_time_series_normalized, pd.Series):
    overall_facilitation_time_series_normalized_smooth = overall_facilitation_time_series_normalized.rolling(window=5, min_periods=1).mean()
else:
    overall_facilitation_time_series_normalized_smooth = overall_facilitation_time_series_normalized

traces.append(go.Scatter(
    x=time_bins[:-1],
    y=overall_facilitation_time_series_normalized_smooth,
    mode='lines',
    name='Overall Facilitation Strategy (All Collections)',
    line=dict(dash='solid'),
    fill='tozeroy',
    fillcolor='rgba(173, 216, 230, 0.3)'
))

for strategy in facilitation_columns:
    facilitation_time_series = df.groupby(pd.cut(df['normalized_time'], bins=time_bins))[strategy].sum()

    total_strategy_count = facilitation_time_series.sum()
    facilitation_time_series_normalized = facilitation_time_series / total_strategy_count

    if isinstance(facilitation_time_series_normalized, pd.Series):
        facilitation_time_series_normalized_smooth = facilitation_time_series_normalized.rolling(window=5, min_periods=1).mean()
    else:
        facilitation_time_series_normalized_smooth = facilitation_time_series_normalized

    traces.append(go.Scatter(
        x=time_bins[:-1],
        y=facilitation_time_series_normalized_smooth,
        mode='lines',
        name=f'{strategy} Strategy (All Collections)',
        line=dict(dash='dot'),
        fill='tozeroy'
    ))

layout = go.Layout(
    title='Normalized Facilitation Strategy Use over Time (0-100)',
    xaxis=dict(title='Normalized Time (0-100)'),
    yaxis=dict(title='Proportion of Total Facilitation Strategies'),
    legend=dict(x=0.05, y=0.95)
)

fig = go.Figure(data=traces, layout=layout)

pio.show(fig)
pio.write_html(fig, file='normalized_overall_and_strategy_facilitation.html', auto_open=True)
