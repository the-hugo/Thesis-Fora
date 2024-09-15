import pandas as pd
import plotly.express as px
import ast

df = pd.read_csv('data/processed_output.csv')

df['Sentence_NER'] = df['Sentence_NER'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

df_exploded = df.explode('Sentence_NER').reset_index(drop=True)  # Reset index to avoid duplicate labels

ner_df = pd.json_normalize(df_exploded['Sentence_NER'])

ner_df['Index in Conversation'] = df_exploded['Index in Conversation']

ner_df = ner_df.dropna(subset=['word', 'entity_group'])

word_count = ner_df.groupby(['entity_group', 'word']).size().reset_index(name='count')

final_df = pd.merge(ner_df[['word', 'entity_group', 'Index in Conversation']],
                    word_count, 
                    on=['word', 'entity_group'],
                    how='left').drop_duplicates()

fig = px.scatter(final_df, 
                 x='Index in Conversation', y='word',
                 size='count', color='entity_group',
                 hover_data=['word', 'entity_group', 'count'],
                 title='NER Visualization with Bubble Diagram',
                 template="plotly_white")

fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
fig.update_layout(
    title_font_size=20,
    xaxis_title='Index in Conversation',
    yaxis_title='Entity',
    showlegend=True,
    legend_title_text='Entity Group',
    template="plotly"
)

fig.show()
