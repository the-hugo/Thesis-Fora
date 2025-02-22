import pandas as pd
import statsmodels.formula.api as smf

# --- Load the Merged Conversational Structure Data ---
merged_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\conversational_structure_with_clusters_with_tests.csv"
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

# --- Compute the Turn Taking Equity Metric (if needed) ---
df_roles['turn_taking_equity'] = 1 - ((df_roles['Gd'] + df_roles['Gc']) / 2)

# --- Recode Facilitator Variables as Binary ---
df_roles['Manager_binary'] = (df_roles['Managers'] > 0).astype(int)
df_roles['Interlocutors_binary'] = (df_roles['Interlocutors'] > 0).astype(int)

# --- Regression Model for Personal Sharing ---
# We include both facilitator roles (as binary) and conversation-level controls.
model_personal_sharing = smf.ols(
    "personal_sharing_ratio ~ Manager_binary + Interlocutors_binary + num_speakers + conversation_length", 
    data=df_roles
).fit()

print("Regression Results for Personal Sharing:")
print(model_personal_sharing.summary())

# --- Regression Model for Personal Story ---
model_personal_story = smf.ols(
    "personal_story_ratio ~ Manager_binary + Interlocutors_binary + num_speakers + conversation_length", 
    data=df_roles
).fit()

print("Regression Results for Personal Story:")
print(model_personal_story.summary())
