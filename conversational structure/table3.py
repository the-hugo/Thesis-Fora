import pandas as pd
import plotly.express as px
from scipy import stats
import itertools
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- Open a log file to save statistical measures ---
log_path = r"C:\path\to\your\log\statistical_measures.txt"
#log_file = open(log_path, "w")

def log(message):
    print(message)
#    log_file.write(message + "\n")

# --- Load the Merged Conversational Structure Data ---
data_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure_with_clusters.csv"
df = pd.read_csv(data_path)
df.rename(columns={"Story tellers": "Storytellers", "Debators": "Debaters"}, inplace=True)

# --- Compute Metrics for Each Conversation ---
# Turn taking equity: computed as 1 - ((Gd + Gc) / 2)
df['turn_taking_equity'] = 1 - ((df['Gd'] + df['Gc']) / 2)
df['personal_sharing_ratio'] = df['personal_sharing'] / df['num_turns']
df['personal_story_ratio'] = df['personal_story'] / df['num_turns']
df['personal_experience_ratio'] = df['personal_experience'] / df['num_turns']

# --- Reshape the Data to Account for Multiple Roles ---
# Instead of assigning a single role to each conversation, we will create a new row for each role
# present in the conversation. We assume the original columns are named "Socializers", "Storytellers", "Debaters".
roles = ['Socializers', 'Storytellers', 'Debaters']
role_mapping = {'Socializers': 'Socializer', 'Storytellers': 'Storyteller', 'Debaters': 'Debater'}

rows = []
for idx, row in df.iterrows():
    for col in roles:
        if row[col] > 0:  # if this role is present in the conversation
            rows.append({
                'role': role_mapping[col],
                'turn_taking_equity': row['turn_taking_equity'],
                'personal_sharing_ratio': row['personal_sharing_ratio'],
                'personal_story_ratio': row['personal_story_ratio'],
                'personal_experience_ratio': row['personal_experience_ratio'],
                'Gc': row['Gc'],
                'Gd': row['Gd'],
                "Number of Turns": row['num_turns'],
                "conversation duration": row['conversation_length'],
                "num_speakers": row['num_speakers']
            })
df_melted = pd.DataFrame(rows)
log("Melted DataFrame created with columns: " + ", ".join(df_melted.columns))

# --- Calculate Average Metrics per Role ---
avg_metrics = df_melted.groupby('role')[["Number of Turns", "conversation duration", 'num_speakers']].mean()
log("\nAverage Metrics per Role:")
log(str(avg_metrics))

# --- Statistical Tests Across Roles ---
unique_roles = df_melted['role'].unique()
log("Roles in melted data: " + ", ".join(unique_roles))

def perform_tests_all(variable_name):
    groups = [df_melted[df_melted['role'] == role][variable_name] for role in unique_roles]
    f_val, p_val = stats.f_oneway(*groups)
    log(f"\nANOVA for {variable_name} across roles:")
    log(f"F-value: {f_val:.20f}, p-value: {p_val:.20f}")
    
    h_stat, p_val_kw = stats.kruskal(*groups)
    log(f"\nKruskal-Wallis test for {variable_name} across roles:")
    log(f"H-statistic: {h_stat:.20f}, p-value: {p_val_kw:.20f}")

for variable in ['personal_sharing_ratio', 'personal_story_ratio', 'personal_experience_ratio', 
                 'turn_taking_equity', 'Gc', 'Gd']:
    perform_tests_all(variable)

# --- Save the Melted DataFrame ---
save_path = r"C:\path\to\your\output\conversational_structure_with_mixed_roles_metrics.csv"
#df_melted.to_csv(save_path, index=False)
log("\nMelted data saved to " + save_path)

# --- Plotting with Plotly ---
def plot_box(variable_name, title, xlabel):
    fig = px.box(
        df_melted,
        x=variable_name,
        y='role',
        color='role',
        title=title,
        labels={variable_name: xlabel, 'role': "Role"}
    )
    fig.update_layout(showlegend=False)
    fig.update_layout(
        font=dict(size=24),
        autosize=False,
        width=1920,
        height=1080
    )
    fig.show()

# Example box plots:
plot_box('personal_sharing_ratio', "Personal Sharing Ratio by Role", "Personal Sharing (per turn)")
plot_box('turn_taking_equity', "Turn Taking Equity by Role", "Turn Taking Equity")

def plot_violin(variable_name, title, xlabel):
    fig = px.violin(
        df_melted,
        x=variable_name,
        y='role',
        color='role',
        box=True,       # overlays a box plot inside the violin
        points='all',   # shows all data points
        title=title,
        labels={variable_name: xlabel, 'role': "Role"}
    )
    fig.update_layout(showlegend=False)
    fig.show()

# Uncomment to generate violin plots:
# plot_violin('personal_sharing_ratio', "Personal Sharing Ratio by Role", "Personal Sharing (per turn)")
# plot_violin('turn_taking_equity', "Turn Taking Equity by Role", "Turn Taking Equity")

# --- Tukey HSD Post-hoc Tests ---
def perform_tukey(variable_name):
    tukey = pairwise_tukeyhsd(endog=df_melted[variable_name],
                              groups=df_melted['role'],
                              alpha=0.05)
    log(f"\nTukey HSD results for {variable_name} across roles:")
    log(str(tukey.summary()))

for variable in ['personal_sharing_ratio', 'personal_story_ratio', 'personal_experience_ratio', 
                 'turn_taking_equity', 'Gc', 'Gd']:
    perform_tukey(variable)

# --- Cohen's d Effect Size for Pairwise Comparisons ---
def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = (((nx - 1) * x.std(ddof=1)**2 + (ny - 1) * y.std(ddof=1)**2) / dof)**0.5
    return (x.mean() - y.mean()) / pooled_std

# log("\nCohen's d effect sizes (Pairwise Comparisons):")
# role_list = sorted(df_melted['role'].unique())
# for variable in ['personal_sharing_ratio', 'personal_story_ratio', 'personal_experience_ratio', 
#                  'turn_taking_equity', 'Gc', 'Gd']:
#     for role1, role2 in itertools.combinations(role_list, 2):
#         data_role1 = df_melted[df_melted['role'] == role1][variable]
#         data_role2 = df_melted[df_melted['role'] == role2][variable]
#         d = cohen_d(data_role1, data_role2)
#         log(f"  For {variable} ({role1} vs. {role2}): d = {d:.20f}")

# # --- Close the log file ---
# log_file.close()
