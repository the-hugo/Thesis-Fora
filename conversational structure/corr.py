import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# File path
target_file = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure_with_clusters_with_tests.csv"

# Read the CSV file
df = pd.read_csv(target_file)

# Define relevant columns
columns = [
    "num_speakers", "conversation_length", "collection_id",
    "Managers", "Interlocutors", "Socializers", "Story tellers", "Debators", "turn_taking_equity", "num_turns", "personal_sharing", "personal_story", "personal_experience", "personal_sharing_ratio", "personal_story_ratio", "personal_experience_ratio"
]

# Compute correlation matrix and p-values
corr_matrix = pd.DataFrame(index=columns, columns=columns)
pval_matrix = pd.DataFrame(index=columns, columns=columns)

for col1 in columns:
    for col2 in columns:
        if col1 == col2:
            corr_matrix.loc[col1, col2] = 1.0
            pval_matrix.loc[col1, col2] = np.nan
        else:
            corr, pval = stats.spearmanr(df[col1], df[col2], nan_policy='omit')
            corr_matrix.loc[col1, col2] = round(corr, 2)
            pval_matrix.loc[col1, col2] = pval

# Add significance notation
def significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

annotated_corr = corr_matrix.astype(str) + pval_matrix.applymap(significance_stars)

# Save to CSV
output_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\correlation_matrix_with_significance.csv"
annotated_corr.to_csv(output_path, index=True)

# Display correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix.astype(float), annot=annotated_corr, fmt="", cmap="coolwarm", center=0)
plt.title("Correlation Matrix with Significance Notation")
plt.show()

print(f"Correlation matrix saved to {output_path}")

