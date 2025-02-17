import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
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

# --- Recode Facilitator Variables as Binary ---
# Since often we have only one facilitator, we recode them as 0 (absent) or 1 (present)
df_roles['Manager_binary'] = (df_roles['Managers'] > 0).astype(int)
df_roles['Interlocutors_binary'] = (df_roles['Interlocutors'] > 0).astype(int)

# --- Explore the Relationship Between Facilitator Roles and Participant Roles ---
# Using the binary facilitator variables in regression models.

# Regression for Socializers
model_socializers_binary = smf.ols("Socializers ~ Manager_binary + Interlocutors_binary + num_speakers + num_turns + conversation_length", 
                                   data=df_roles).fit()
print("\nRegression Results for Socializers (using binary facilitator variables):")
print(model_socializers_binary.summary())

# Regression for Story tellers (using Q() for the column with a space)
model_storytellers_binary = smf.ols("Q('Story tellers') ~ Manager_binary + Interlocutors_binary + num_speakers + num_turns + conversation_length", 
                                    data=df_roles).fit()
print("\nRegression Results for Story tellers (using binary facilitator variables):")
print(model_storytellers_binary.summary())

# Regression for Debators
model_debators_binary = smf.ols("Debators ~ Manager_binary + Interlocutors_binary + num_speakers + num_turns + conversation_length", 
                                data=df_roles).fit()
print("\nRegression Results for Debators (using binary facilitator variables):")
print(model_debators_binary.summary())

# --- Save the Updated DataFrame with Binary Variables ---
save_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure_with_clusters_with_tests_binary.csv"
df_roles.to_csv(save_path, index=False)
print("\nData saved to", save_path)

# exclude non numerical values in df_roles
df_roles = df_roles.select_dtypes(include=[np.number])
# exclude columns conversation:id, collection_id, personal_experience, personal_story, Gd, Gc
df_roles = df_roles.drop(columns=['conversation_id', 'collection_id', 'personal_experience', 'personal_story', 'Gd', 'Gc'])

# --- Compute the Correlation Matrix ---
corr_matrix = df_roles.corr()

# Print the correlation matrix to the console
print("Correlation Matrix:")
print(corr_matrix)

# --- Visualize the Correlation Matrix with a Heatmap ---
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of df_roles')
plt.show()
