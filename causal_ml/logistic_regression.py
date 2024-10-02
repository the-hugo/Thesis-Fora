import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from causalml.inference.meta import LRSRegressor
from sklearn.model_selection import cross_val_score

# Load your dataset (assuming it's a CSV file, adjust the path accordingly)
df = pd.read_csv('your_dataset.csv')

# 1. Data Preparation

# Create separate dataframes for facilitators and participants
facilitator_df = df[df['is_facilitator'] == True]  # Only facilitator rows
participant_df = df[df['is_facilitator'] == False]  # Only participant rows

# Feature engineering: add turn duration and word count
df['turn_duration'] = df['audio_end_offset'] - df['audio_start_offset']
df['word_count'] = df['words'].apply(lambda x: len(str(x).split()))

# 2. Use SpeakerTurn and link the most recent facilitator's strategy to the next participant's turn

# Sort by conversation_id and SpeakerTurn to ensure proper order of conversation
df = df.sort_values(by=['conversation_id', 'SpeakerTurn'])

# Forward-fill facilitation strategies within each conversation and only fill when the previous speaker was a facilitator
df['prev_facilitation_strategy'] = df.groupby('conversation_id')['facilitation_strategy'].ffill()
df['prev_speaker_is_facilitator'] = df.groupby('conversation_id')['is_facilitator'].shift(1)

# Filter to get rows where the previous speaker was a facilitator and the current speaker is a participant
df_participants_with_strategy = df[(df['is_facilitator'] == False) & (df['prev_speaker_is_facilitator'] == True)].copy()

# Remove any rows where the facilitation strategy wasn't filled
df_participants_with_strategy = df_participants_with_strategy.dropna(subset=['prev_facilitation_strategy'])

# 3. Propensity Score Estimation

# Define covariates, treatment, and outcome
X = df_participants_with_strategy[['turn_duration', 'word_count']]  # Covariates
treatment = df_participants_with_strategy['prev_facilitation_strategy']  # Treatment from previous facilitator
personal_sharing = df_participants_with_strategy['personal_sharing']  # Outcome (participant's sharing behavior)

# Split data into training and testing sets for validation
X_train, X_test, treatment_train, treatment_test = train_test_split(X, treatment, test_size=0.2, random_state=42)

# Standardize the covariates
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit a logistic regression model to estimate propensity scores
logreg = LogisticRegression()
logreg.fit(X_train_scaled, treatment_train)

# Add the predicted propensity scores to the DataFrame
df_participants_with_strategy['propensity_score'] = logreg.predict_proba(X)[:, 1]

# Check the distribution of propensity scores (optional)
df_participants_with_strategy['propensity_score'].hist()

# 4. Double Machine Learning (DML) Outcome Modeling

# Initialize the LRSRegressor for Double Machine Learning
lr = LRSRegressor()

# Estimate the Average Treatment Effect (ATE) using Double Machine Learning
te, lb, ub = lr.estimate_ate(X=X, treatment=treatment, y=personal_sharing)

print(f"Estimated Treatment Effect on Personal Sharing: {te}")
print(f"95% Confidence Interval: ({lb}, {ub})")

# 5. Validation Using Cross-Validation

# Perform cross-validation to ensure model robustness
cv_scores = cross_val_score(lr.model_t, X, personal_sharing, cv=5)

print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {cv_scores.mean()}")
