import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File path for the merged CSV with role compositions
file_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure_with_clusters.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Define the role columns (adjust if necessary)
role_columns = ['Managers', 'Interlocutors', 'Socializers', 'Story tellers', 'Debators']

# Display descriptive statistics for the role counts
role_stats = df[role_columns].describe()
print("Descriptive Statistics for Role Counts:")
print(role_stats)

# ----------------------
# Visualization Section
# ----------------------

# 1. Bar Chart: Total counts of each role across all conversations
total_counts = df[role_columns].sum()
plt.figure(figsize=(10, 6))
sns.barplot(x=total_counts.index, y=total_counts.values, palette='viridis')
plt.title("Total Counts of Each Role Across All Conversations")
plt.ylabel("Count")
plt.xlabel("Role")
plt.tight_layout()
plt.show()

# 2. Stacked Bar Chart: Composition of roles per conversation
# Sorting by conversation_id for clarity (or sort by total counts if preferred)
df_sorted = df.sort_values(by='conversation_id')

plt.figure(figsize=(12, 8))
bottom = None
for role in role_columns:
    if bottom is None:
        plt.bar(df_sorted['conversation_id'], df_sorted[role], label=role)
        bottom = df_sorted[role].values
    else:
        plt.bar(df_sorted['conversation_id'], df_sorted[role], bottom=bottom, label=role)
        bottom += df_sorted[role].values

plt.xticks(rotation=90)
plt.xlabel("Conversation ID")
plt.ylabel("Count")
plt.title("Role Distribution per Conversation")
plt.legend(title="Role")
plt.tight_layout()
plt.show()

# 3. Box Plots: Distribution of counts for each role across conversations
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[role_columns], palette='Set2')
plt.title("Distribution of Role Counts Across Conversations")
plt.ylabel("Count")
plt.xlabel("Role")
plt.tight_layout()
plt.show()
