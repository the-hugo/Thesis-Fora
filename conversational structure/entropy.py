import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# File path to the CSV file with the role counts per conversation
csv_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure_with_clusters.csv"

# Read the CSV file
df = pd.read_csv(csv_path)

# Define all the role columns
role_columns = ['Managers', 'Interlocutors', 'Socializers', 'Story tellers', 'Debators']

# --- Compute the Correlation Matrix for Role Counts ---
corr_matrix = df[role_columns].corr()
print("Correlation matrix of role counts:")
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation between Role Counts")
plt.tight_layout()
plt.show()

# --- Compute the Co-Occurrence Matrix ---
# Convert role counts to binary presence (1 if count > 0, else 0)
binary_presence = df[role_columns].applymap(lambda x: 1 if x > 0 else 0)
co_occurrence = binary_presence.T.dot(binary_presence)
print("Co-occurrence matrix (number of conversations where roles appear together):")
print(co_occurrence)

plt.figure(figsize=(8, 6))
sns.heatmap(co_occurrence, annot=True, cmap='YlGnBu', fmt="d")
plt.title("Co-occurrence of Roles Across Conversations")
plt.tight_layout()
plt.show()

# --- Test Significance of Co-occurrence ---
# Create a DataFrame to store p-values for the chi-square tests
p_values = pd.DataFrame(np.ones((len(role_columns), len(role_columns))),
                        index=role_columns, columns=role_columns)

# Loop over each unique pair of roles to compute a chi-square test for independence
for i, role1 in enumerate(role_columns):
    for j, role2 in enumerate(role_columns):
        if i < j:
            # Build a contingency table:
            # [both present, role1 only]
            # [role2 only, neither present]
            both = ((binary_presence[role1] == 1) & (binary_presence[role2] == 1)).sum()
            only_role1 = ((binary_presence[role1] == 1) & (binary_presence[role2] == 0)).sum()
            only_role2 = ((binary_presence[role1] == 0) & (binary_presence[role2] == 1)).sum()
            neither = ((binary_presence[role1] == 0) & (binary_presence[role2] == 0)).sum()
            
            contingency_table = np.array([[both, only_role1],
                                          [only_role2, neither]])
            # Perform the chi-square test
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            p_values.loc[role1, role2] = p
            p_values.loc[role2, role1] = p

print("P-values matrix for chi-square test of independence (co-occurrence significance):")
print(p_values)

plt.figure(figsize=(8, 6))
sns.heatmap(p_values, annot=True, cmap='viridis_r', fmt=".3f")
plt.title("P-values for Co-occurrence Significance (Chi-square Test)")
plt.tight_layout()
plt.show()
