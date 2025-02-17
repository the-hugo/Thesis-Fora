import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

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
personal_story_groups = [
    df_roles[df_roles['role_group'] == group]['personal_story'] for group in role_groups
]

# One-way ANOVA for personal_story
f_val_pst, p_val_pst = stats.f_oneway(*personal_story_groups)
print("\nANOVA for personal_story across role groups:")
print("F-value: {:.4f}, p-value: {:.4f}".format(f_val_pst, p_val_pst))

# Kruskal-Wallis test for personal_story
h_stat_pst, p_val_kw_pst = stats.kruskal(*personal_story_groups)
print("\nKruskal-Wallis test for personal_story across role groups:")
print("H-statistic: {:.4f}, p-value: {:.4f}".format(h_stat_pst, p_val_kw_pst))

# --- Test Relationship between Role Group and Turn Taking Equity ---
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

# --- Post Hoc Pairwise Comparisons for Turn Taking Equity ---
# Using Tukey's HSD test
tukey_results = pairwise_tukeyhsd(
    endog=df_roles['turn_taking_equity'], 
    groups=df_roles['role_group'], 
    alpha=0.05
)
print("\nTukey HSD results for turn taking equity:")
print(tukey_results)

# Optionally, plot the Tukey HSD results.
tukey_results.plot_simultaneous(ylabel='Role Group', xlabel='Difference in Turn Taking Equity')
plt.title('Tukey HSD Test for Turn Taking Equity')
plt.tight_layout()
plt.show()

# --- Post Hoc Pairwise Comparisons for Personal Story ---
# Using Tukey's HSD test for personal_story if needed
tukey_results_pst = pairwise_tukeyhsd(
    endog=df_roles['personal_story'], 
    groups=df_roles['role_group'], 
    alpha=0.05
)
print("\nTukey HSD results for personal_story:")
print(tukey_results_pst)

# Optionally, plot the Tukey HSD results for personal_story.
tukey_results_pst.plot_simultaneous(ylabel='Role Group', xlabel='Difference in Personal Story')
plt.title('Tukey HSD Test for Personal Story')
plt.tight_layout()
plt.show()

# --- Calculate Cohen's d for the Difference in Turn Taking Equity ---
# Between Manager and Interlocutor groups
manager_data = df_roles[df_roles['role_group'] == 'Manager']['turn_taking_equity']
interlocutor_data = df_roles[df_roles['role_group'] == 'Interlocutor']['turn_taking_equity']

mean_manager = manager_data.mean()
mean_interlocutor = interlocutor_data.mean()
std_manager = manager_data.std()
std_interlocutor = interlocutor_data.std()

n_manager = len(manager_data)
n_interlocutor = len(interlocutor_data)

# Pooled standard deviation calculation
pooled_std = np.sqrt(((n_manager - 1) * std_manager**2 + (n_interlocutor - 1) * std_interlocutor**2) / (n_manager + n_interlocutor - 2))

cohens_d = (mean_interlocutor - mean_manager) / pooled_std

print("\nCohen's d for the difference in turn taking equity between Interlocutor and Manager:")
print("{:.4f}".format(cohens_d))

# --- Save the Updated DataFrame ---
save_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure_with_clusters_with_tests.csv"
df_roles.to_csv(save_path, index=False)
print("\nData saved to", save_path)
