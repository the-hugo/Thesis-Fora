import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# --- Load the Merged Conversational Structure Data ---
merged_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure_with_clusters.csv"
df = pd.read_csv(merged_path)

# --- Define Role Group for Each Conversation ---
def assign_role(row):
    # Only consider conversations that have at least one facilitator role.
    if row['Managers'] > 0 and row['Interlocutors'] > 0:
        return "Mixed Roles"
    elif row['Managers'] > 0:
        return "Manager"
    elif row['Interlocutors'] > 0:
        return "Interlocutor"
    else:
        return None  # if neither is present

# Create a new column for role group and drop rows without any facilitator roles.
df['role_group'] = df.apply(assign_role, axis=1)
df_roles = df[df['role_group'].notna()].copy()
print("Columns in df_roles:", df_roles.columns)

# --- Compute the Turn Taking Equity Metric ---
# Turn taking equity is defined as: 1 - ((Gd + Gc) / 2)
df_roles['turn_taking_equity'] = 1 - ((df_roles['Gd'] + df_roles['Gc']) / 2)

# --- Statistical Tests ---

# List the unique role groups (e.g., "Manager", "Interlocutor", "Mixed Roles")
role_groups = df_roles['role_group'].unique()
print("Role groups found:", role_groups)

# --- Test Relationship between Role Group and Personal Sharing ---

# Group personal_sharing values by role_group
personal_sharing_groups = [
    df_roles[df_roles['role_group'] == group]['personal_sharing'] for group in role_groups
]

# One-way ANOVA for personal_sharing
f_val_ps, p_val_ps = stats.f_oneway(*personal_sharing_groups)
print("\nANOVA for personal_sharing across role groups:")
print("F-value: {:.4f}, p-value: {:.4f}".format(f_val_ps, p_val_ps))

# Kruskal-Wallis test (non-parametric) for personal_sharing
h_stat_ps, p_val_kw_ps = stats.kruskal(*personal_sharing_groups)
print("\nKruskal-Wallis test for personal_sharing across role groups:")
print("H-statistic: {:.4f}, p-value: {:.4f}".format(h_stat_ps, p_val_kw_ps))

# --- Test Relationship between Role Group and Personal Story ---

# Group personal_story values by role_group
personal_story_groups = [
    df_roles[df_roles['role_group'] == group]['personal_story'] for group in role_groups
]

# One-way ANOVA for personal_story
f_val_story, p_val_story = stats.f_oneway(*personal_story_groups)
print("\nANOVA for personal_story across role groups:")
print("F-value: {:.4f}, p-value: {:.4f}".format(f_val_story, p_val_story))

# Kruskal-Wallis test (non-parametric) for personal_story
h_stat_story, p_val_kw_story = stats.kruskal(*personal_story_groups)
print("\nKruskal-Wallis test for personal_story across role groups:")
print("H-statistic: {:.4f}, p-value: {:.4f}".format(h_stat_story, p_val_kw_story))

# --- Test Relationship between Role Group and Turn Taking Equity ---

# Group turn_taking_equity values by role_group
equity_groups = [
    df_roles[df_roles['role_group'] == group]['turn_taking_equity'] for group in role_groups
]

# One-way ANOVA for turn_taking_equity
f_val_te, p_val_te = stats.f_oneway(*equity_groups)
print("\nANOVA for turn_taking_equity across role groups:")
print("F-value: {:.4f}, p-value: {:.4f}".format(f_val_te, p_val_te))

# Kruskal-Wallis test for turn_taking_equity
h_stat_te, p_val_kw_te = stats.kruskal(*equity_groups)
print("\nKruskal-Wallis test for turn_taking_equity across role groups:")
print("H-statistic: {:.4f}, p-value: {:.4f}".format(h_stat_te, p_val_kw_te))

# --- Save the Updated DataFrame ---
save_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure_with_clusters_with_tests.csv"
df_roles.to_csv(save_path, index=False)
print("\nData saved to", save_path)

# --- Plot Histograms for the Role Columns ---

# Define role columns
role_columns = ['Managers', 'Interlocutors', 'Socializers', 'Story tellers', 'Debators']

plt.figure(figsize=(15, 10))
for i, col in enumerate(role_columns, 1):
    plt.subplot(2, 3, i)
    # Define bins based on the maximum count for the role (plus one to include the max)
    plt.hist(df[col], bins=range(0, df[col].max() + 2), edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

roles_to_plot = ['Manager', 'Interlocutor', 'Mixed Roles']
df_plot = df_roles[df_roles['role_group'].isin(roles_to_plot)]

plt.figure(figsize=(8, 6))
sns.boxplot(x='personal_story', y='role_group', data=df_plot, palette='pastel')
plt.title("Personal Story by Role Group")
plt.xlabel("Personal Story")
plt.ylabel("Role Group")
plt.tight_layout()
plt.show()

# --- (Optional) Create Horizontal Box Plots for Personal Sharing and Turn Taking Equity by Role Group ---
# Uncomment these sections if you wish to view these plots as well.

# # 1. Box Plot for Personal Sharing
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='personal_sharing', y='role_group', data=df_roles, orient='h')
# plt.title("Personal Sharing by Role Group")
# plt.xlabel("Personal Sharing")
# plt.ylabel("Role Group")
# plt.tight_layout()
# plt.show()

# # 2. Box Plot for Turn Taking Equity
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='turn_taking_equity', y='role_group', data=df_roles, orient='h')
# plt.title("Turn Taking Equity by Role Group")
# plt.xlabel("Turn Taking Equity (1 - Mean Gini)")
# plt.ylabel("Role Group")
# plt.tight_layout()
# plt.show()
