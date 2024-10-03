from econml.grf import CausalForest
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load and prepare your dataset
df = pd.read_pickle(r'C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\output\umap\data_nv-embed_processed_output.pkl')
df = df[df['annotated']].copy()

# Create personal_sharing and facilitation_strategy variables
df['personal_sharing'] = df[['Personal story', 'Personal experience']].max(axis=1)
df['facilitation_strategy'] = df[['Express affirmation', 'Specific invitation', 'Provide example', 'Open invitation', 'Make connections', 'Express appreciation', 'Follow up question']].max(axis=1)
df['total_facilitation_strategies'] = df.groupby('conversation_id')['facilitation_strategy'].transform('sum')

# Select participant data
df_participants = df[df['is_fac'] == False]

# Treatment (total facilitation strategies) and outcome (personal sharing)
T = df_participants['total_facilitation_strategies'].values.reshape(-1, 1)  # Reshape treatment to 2D
Y = df_participants['personal_sharing'].values.reshape(-1, 1)  # Reshape outcome to 2D

# Covariates (duration, word_count, SpeakerTurn)
X = df_participants[['duration', 'word_count', 'SpeakerTurn']]

# Train-test split for cross-validation
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=42)

# Reshape the treatment and outcome arrays to ensure they are 2D
T_train = T_train.reshape(-1, 1)
T_test = T_test.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

# Initialize the Causal Forest model
causal_forest = CausalForest()

# Fit the model
causal_forest.fit(Y_train, T_train, X_train)

# Estimate the treatment effect for test data
treatment_effects = causal_forest.effect(X_test)

# Average Treatment Effect (ATE)
ate_cf = np.mean(treatment_effects)
print(f"Average Treatment Effect (ATE) using Causal Forest: {ate_cf}")

# Confidence Interval for the ATE
lb_cf, ub_cf = causal_forest.effect_interval(X_test)
print(f"Confidence Interval for ATE using Causal Forest: [{lb_cf.mean()}, {ub_cf.mean()}]")

# Plot the heterogeneous treatment effects
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(T_test, treatment_effects)
plt.title('Heterogeneous Treatment Effect of Facilitation Strategies on Personal Sharing')
plt.xlabel('Total Facilitation Strategies')
plt.ylabel('Estimated Treatment Effect')
plt.show()
