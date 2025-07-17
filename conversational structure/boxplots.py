import pandas as pd
import plotly.express as px
from scipy import stats
import itertools
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- Open a log file to save statistical measures ---
log_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\statistical_measures.txt"
log_file = open(log_path, "w")

def log(message):
    print(message)
    log_file.write(message + "\n")

# --- Load the Merged Conversational Structure Data ---
merged_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure_with_clusters.csv"
df = pd.read_csv(merged_path)

# --- Define Role Group for Each Conversation ---
def assign_role(row):
    if row['Managers'] > 0 and row['Interlocutors'] > 0:
        return "Mixed Roles"
    elif row['Managers'] > 0:
        return "Manager"
    elif row['Interlocutors'] > 0:
        return "Interlocutor"
    else:
        return None

df['role_group'] = df.apply(assign_role, axis=1)
df_roles = df[df['role_group'].notna()].copy()
log("Columns in df_roles: " + ", ".join(df_roles.columns))

# --- Compute the Turn Taking Equity Metric ---
df_roles['turn_taking_equity'] = 1 - ((df_roles['Gd'] + df_roles['Gc']) / 2)

# --- Create Ratio Variables ---
df_roles['personal_story_ratio'] = df_roles['personal_story'] / df_roles['num_turns']
df_roles['personal_sharing_ratio'] = df_roles['personal_sharing'] / df_roles['num_turns']
df_roles['personal_experience_ratio'] = df_roles['personal_experience'] / df_roles['num_turns']

# --- Statistical Tests for All Role Groups ---
role_groups = df_roles['role_group'].unique()
log("Role groups found: " + ", ".join(role_groups))

def perform_tests_all(variable_name):
    groups = [df_roles[df_roles['role_group'] == group][variable_name] for group in role_groups]
    f_val, p_val = stats.f_oneway(*groups)
    log(f"\nANOVA for {variable_name} across role groups:")
    log(f"F-value: {f_val:.20f}, p-value: {p_val:.20f}")
    
    h_stat, p_val_kw = stats.kruskal(*groups)
    log(f"\nKruskal-Wallis test for {variable_name} across role groups:")
    log(f"H-statistic: {h_stat:.20f}, p-value: {p_val_kw:.20f}")

# --- Perform tests for the existing variables ---
for variable in ['personal_sharing_ratio', 'personal_story_ratio', 'personal_experience_ratio', 'turn_taking_equity']:
    perform_tests_all(variable)

# --- Perform tests for Gc and Gd ---
for variable in ['Gc', 'Gd']:
    perform_tests_all(variable)

# --- Save the Updated DataFrame ---
save_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure_with_clusters_with_tests.csv"
df_roles.to_csv(save_path, index=False)
log("\nData saved to " + save_path)

# --- Filter Data to Only Include Manager and Interlocutor ---
df_mi = df_roles[df_roles['role_group'].isin(['Manager', 'Interlocutor'])].copy()

# Set role_group as an ordered categorical variable so that "Manager" comes first.
df_mi['role_group'] = pd.Categorical(df_mi['role_group'],
                                     categories=['Manager', 'Interlocutor'],
                                     ordered=True)

# Define category order for Plotly visualizations.
category_order = {'role_group': ['Manager', 'Interlocutor']}

# --- Plotting with Plotly ---
def plot_box(variable_name, title, xlabel):
    fig = px.box(
        df_mi,
        x=variable_name,
        y='role_group',
        color='role_group',
        title=title,
        labels={variable_name: xlabel, 'role_group': "Role Group"},
        category_orders=category_order
    )
    fig.update_layout(showlegend=False)
    fig.update_layout(
        font=dict(size=24),
        autosize=False,
        width=1920,
        height=1080
    )
    fig.show()

# Uncomment the lines below to generate box plots:
# plot_box('personal_story_ratio', "Personal Story Ratio by Role", "Personal Story (per turn)")
plot_box('personal_sharing_ratio', "Personal Sharing Ratio by Role", "Personal Sharing (per turn)")
# plot_box('personal_experience_ratio', "Personal Experience Ratio by Role", "Personal Experience (per turn)")
plot_box('turn_taking_equity', "Turn Taking Equity by Role", "Turn Taking Equity")


def plot_violin(variable_name, title, xlabel):
    fig = px.violin(
        df_mi,
        x=variable_name,
        y='role_group',
        color='role_group',
        box=True,       # overlays a box plot inside the violin
        points='all',   # shows all data points
        title=title,
        labels={variable_name: xlabel, 'role_group': "Role Group"},
        category_orders=category_order
    )
    fig.update_layout(showlegend=False)
    fig.show()

# Uncomment the lines below to generate violin plots:
# plot_violin('personal_story_ratio', "Personal Story Ratio by Role Group", "Personal Story (per turn)")
# plot_violin('personal_sharing_ratio', "Personal Sharing Ratio by Role Group", "Personal Sharing (per turn)")
# plot_violin('personal_experience_ratio', "Personal Experience Ratio by Role Group", "Personal Experience (per turn)")
# plot_violin('turn_taking_equity', "Turn Taking Equity by Role Group", "Turn Taking Equity")
# Similarly, for Gc and Gd:
# plot_violin('Gc', "Gc by Role Group", "Gc")
# plot_violin('Gd', "Gd by Role Group", "Gd")

# --- Statistical Tests for Managers vs. Interlocutors ---
def perform_tests(variable_name):
    groups = [df_mi[df_mi['role_group'] == group][variable_name] for group in ['Manager', 'Interlocutor']]
    f_val, p_val = stats.f_oneway(*groups)
    log(f"\nANOVA for {variable_name} (Managers vs. Interlocutors):")
    log(f"F-value: {f_val:.20f}, p-value: {p_val:.20f}")
    
    h_stat, p_val_kw = stats.kruskal(*groups)
    log(f"\nKruskal-Wallis test for {variable_name} (Managers vs. Interlocutors):")
    log(f"H-statistic: {h_stat:.20f}, p-value: {p_val_kw:.20f}")

# --- Perform tests for the existing variables ---
for variable in ['personal_sharing_ratio', 'personal_story_ratio', 'personal_experience_ratio', 'turn_taking_equity']:
    perform_tests(variable)

# --- Perform tests for Gc and Gd ---
for variable in ['Gc', 'Gd']:
    perform_tests(variable)

# --- Tukey HSD Post-hoc Test on the Filtered Data ---
def perform_tukey(variable_name):
    tukey = pairwise_tukeyhsd(endog=df_mi[variable_name],
                              groups=df_mi['role_group'],
                              alpha=0.05)
    log(f"\nTukey HSD results for {variable_name} (Managers vs. Interlocutors):")
    # Convert the summary table to a string and log it.
    log(str(tukey.summary()))

# --- Perform Tukey HSD for the existing variables ---
for variable in ['personal_sharing_ratio', 'personal_story_ratio', 'personal_experience_ratio', 'turn_taking_equity']:
    perform_tukey(variable)

# --- Perform Tukey HSD for Gc and Gd ---
for variable in ['Gc', 'Gd']:
    perform_tukey(variable)

# --- Cohen's d Effect Size ---
def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = (((nx - 1) * x.std(ddof=1)**2 + (ny - 1) * y.std(ddof=1)**2) / dof)**0.5
    # Compute d as (Manager_mean - Interlocutor_mean); a positive value indicates Managers have higher values.
    return (x.mean() - y.mean()) / pooled_std

log("\nCohen's d effect sizes (Managers vs. Interlocutors):")
for variable in ['personal_sharing_ratio', 'personal_story_ratio', 'personal_experience_ratio', 'turn_taking_equity']:
    data_manager = df_mi[df_mi['role_group'] == 'Manager'][variable]
    data_interlocutor = df_mi[df_mi['role_group'] == 'Interlocutor'][variable]
    d = cohen_d(data_manager, data_interlocutor)
    log(f"  For {variable}: d = {d:.20f}")

# --- Cohen's d for Gc and Gd ---
for variable in ['Gc', 'Gd']:
    data_manager = df_mi[df_mi['role_group'] == 'Manager'][variable]
    data_interlocutor = df_mi[df_mi['role_group'] == 'Interlocutor'][variable]
    d = cohen_d(data_manager, data_interlocutor)
    log(f"  For {variable}: d = {d:.20f}")

# --- Save Boxplot Statistics (including the mean) ---
def save_boxplot_stats(variable_name):
    for group in ['Manager', 'Interlocutor']:
        data = df_mi[df_mi['role_group'] == group][variable_name]
        mean_val = data.mean()
        q1 = data.quantile(0.25)
        median = data.median()
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        # Determine whiskers based on the 1.5 * IQR rule
        lower_whisker = data[data >= q1 - 1.5 * iqr].min()
        upper_whisker = data[data <= q3 + 1.5 * iqr].max()
        # Identify outliers
        outliers = data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)]
        
        log(f"\nBoxplot statistics for {variable_name} in group {group}:")
        log(f"  Mean: {mean_val:.4f}")
        log(f"  Q1: {q1:.4f}")
        log(f"  Median: {median:.4f}")
        log(f"  Q3: {q3:.4f}")
        log(f"  IQR: {iqr:.4f}")
        log(f"  Lower Whisker: {lower_whisker:.4f}")
        log(f"  Upper Whisker: {upper_whisker:.4f}")
        log(f"  Outliers: {outliers.tolist()}")

# --- Save boxplot stats for the existing variables ---
for variable in ['personal_sharing_ratio', 'personal_story_ratio', 'personal_experience_ratio', 'turn_taking_equity']:
    save_boxplot_stats(variable)

# --- Save boxplot stats for Gc and Gd ---
for variable in ['Gc', 'Gd']:
    save_boxplot_stats(variable)

# --- Close the log file ---
log_file.close()
