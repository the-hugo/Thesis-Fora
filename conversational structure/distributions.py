import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# File path for the merged CSV with role compositions
file_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure_with_clusters.csv"

# Read the CSV file
df = pd.read_csv(file_path)
df.rename(columns={"Story tellers": "Storytellers"}, inplace=True)

# Define the role columns
role_columns = ['Managers', 'Interlocutors', 'Socializers', 'Storytellers', 'Debators']

# ----------------------
# 1. Bar Chart: Total counts of each role across all conversations
# ----------------------
total_counts = df[role_columns].sum().reset_index()
total_counts.columns = ['Role', 'Count']

# Define a palette for roles
palette = {
    'Interlocutors': '#00A4EB',
    'Managers': '#00A4EB',
    'Debators': '#FFC600',
    'Storytellers': '#FFC600',
    'Socializers': '#FFC600'
}

fig1 = px.bar(total_counts,
              x='Role',
              y='Count',
              color='Role',
              color_discrete_map=palette,
              title="Total Counts of Each Role Across Fora Corpus")
fig1.update_layout(xaxis_title="Role", yaxis_title="Count")
fig1.show()

# ----------------------
# 2. Stacked Bar Chart: Composition of roles per conversation
# ----------------------
# Sort by conversation_id for clarity
df_sorted = df.sort_values(by='conversation_id')

# Create traces for each role
fig2 = go.Figure()
for role in role_columns:
    fig2.add_trace(go.Bar(
        x=df_sorted['conversation_id'],
        y=df_sorted[role],
        name=role,
        marker_color=palette.get(role, None)
    ))

fig2.update_layout(barmode='stack',
                   title="Role Distribution per Conversation",
                   xaxis_title="Conversation ID",
                   yaxis_title="Count")
fig2.show()

# ----------------------
# 3. Box Plots: Distribution of counts for each role across conversations
# ----------------------
# Melt the dataframe to have a long-form version suitable for box plots
melted = df.melt(value_vars=role_columns, var_name='Role', value_name='Count')

# Define an alternative palette for box plots if needed
palette_box = {
    'Interlocutors': '#00A5EC',
    'Managers': '#00A5EC',
    'Debators': '#FFC500',
    'Storytellers': '#FFC500',
    'Socializers': '#FFC500'
}

fig3 = px.box(melted,
              x='Role',
              y='Count',
              color='Role',
              color_discrete_map=palette_box,
              title="Distribution of Role Counts Across Conversations")
fig3.update_layout(xaxis_title="Role", yaxis_title="Count")
fig3.show()
