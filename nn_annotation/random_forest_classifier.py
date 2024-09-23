import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
import umap
from sklearn.preprocessing import StandardScaler

# Helper function to evaluate model performance
def evaluate_model(y_true, y_pred, y_pred_prob, label, threshold=0.5):
    """Evaluate model performance using accuracy, precision, recall, f1-score, AUC-ROC, and confusion matrix."""
    # Apply threshold tuning
    y_pred_adjusted = (y_pred_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred_adjusted)
    precision = precision_score(y_true, y_pred_adjusted)
    recall = recall_score(y_true, y_pred_adjusted)
    f1 = f1_score(y_true, y_pred_adjusted)
    auc_roc = roc_auc_score(y_true, y_pred_prob)
    
    print(f"\nEvaluation metrics for {label} (Threshold: {threshold}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_adjusted)
    plot_confusion_matrix(cm, label)

def plot_confusion_matrix(cm, label):
    """Plot the confusion matrix."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {label}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Load data function
def load_data(input_path):
    """Load data from a pickle file."""
    print(f"Loading data from {input_path}")
    return pd.read_pickle(input_path)

# Split data
def split_data(df):
    """Split the data into annotated and unannotated subsets."""
    if 'annotated' not in df.columns:
        raise ValueError("The DataFrame does not contain the 'annotated' column.")
    df_annotated = df[df['annotated']].copy()
    df_unannotated = df[~df['annotated']].copy()
    return df_annotated, df_unannotated

# Preprocess data with UMAP if used earlier
def preprocess_data(df, n_components=100):
    """Preprocess the data by converting columns to numeric and applying UMAP for dimensionality reduction."""
    df['SpeakerTurn'] = pd.to_numeric(df['SpeakerTurn'])
    df['audio_start_offset'] = pd.to_numeric(df['audio_start_offset'])
    df['audio_end_offset'] = pd.to_numeric(df['audio_end_offset'])
    df['is_fac'] = pd.to_numeric(df['is_fac']).astype(int)
    df['cofacilitated'] = pd.to_numeric(df['cofacilitated']).astype(int)

    # Stack the embeddings into a 2D array
    embeddings = np.vstack(df['Latent-Attention_Embedding'].values)

    # Apply UMAP to reduce dimensionality of embeddings
    print(f"Original embedding shape: {embeddings.shape}")
    
    # Standardize embeddings before UMAP
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Reduce dimensionality with UMAP
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embeddings_reduced = reducer.fit_transform(embeddings_scaled)

    print(f"Reduced embedding shape: {embeddings_reduced.shape}")

    # Combine other features with the reduced embeddings
    other_features = df[['SpeakerTurn', 'audio_start_offset', 'audio_end_offset', 'is_fac', 'cofacilitated']].values
    X = np.hstack([other_features, embeddings_reduced])

    print("Shape of X after UMAP:", X.shape)

    return X

# Split into train and test sets
def split_train_test(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets with stratification."""
    X = np.array(X)  # Ensure it's a numpy array
    y = np.array(y)
    
    # Stratified split to maintain class distribution in train and test sets
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def main(input_path, model_save_path, random_state=42, use_xgboost=False):
    """Main function to load data, preprocess using UMAP, train, and evaluate the model."""
    # Load and preprocess data
    df = load_data(input_path)
    df_annotated, df_unannotated = split_data(df)
    
    # Preprocess data to get reduced features and embeddings
    X = preprocess_data(df_annotated, n_components=100)  # UMAP reduces embeddings to 100 dimensions
    
    # Extract the target labels (binary classification)
    target = ['Personal story', 'Personal experience', 'Express affirmation', 'Specific invitation', 
              'Provide example', 'Open invitation', 'Make connections', 'Express appreciation', 'Follow up question']
    
    # Initialize a dictionary to store the models
    models = {}
    
    # Train a separate binary classifier for each target variable
    for label in tqdm(target, desc="Training models"):
        print(f"\nTraining model for {label}...")

        # Extract y for the current label
        y = df_annotated[label].values
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        
        # Handle class imbalance using SMOTE
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # Use XGBoost or Random Forest

        # Use GridSearchCV to find optimal hyperparameters for Random Forest
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        clf = GridSearchCV(RandomForestClassifier(random_state=random_state, class_weight={0: 1, 1: 5}),
        param_grid, scoring='precision', cv=3)
        
        # Train the model
        clf.fit(X_train_res, y_train_res)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]  # Probability for the positive class
        
        # Evaluate the model with threshold tuning (e.g., setting threshold to 0.5)
        evaluate_model(y_test, y_pred, y_pred_prob, label, threshold=0.5)

        # Save the trained model for this label
        models[label] = clf

    # Save all models to disk
    with open(model_save_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"All models saved to {model_save_path}")

if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\data_nv-embed_processed_output.pkl"
    model_save_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\random_forest_models_smoteenn.pkl"
    
    # Call main function, set use_xgboost to True if you want to try XGBoost
    main(input_path, model_save_path, random_state=42, use_xgboost=False)
