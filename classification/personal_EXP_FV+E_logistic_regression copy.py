import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(input_path):
    """
    Load data from a pickle file.

    Parameters:
    - input_path (str): Path to the input pickle file.

    Returns:
    - pd.DataFrame: Loaded data as a DataFrame.
    """
    print(f"Loading data from {input_path}")
    return pd.read_pickle(input_path)


def preprocess_data(data):
    """
    Preprocess the input data.

    Parameters:
    - data (pd.DataFrame): Input data.

    Returns:
    - np.ndarray: Feature matrix.
    - np.ndarray: Target array.
    """
    # Drop rows with missing embeddings
    data = data.dropna(subset=["Latent-Attention_Embedding"])

    # Convert embeddings to numpy arrays
    data["Latent-Attention_Embedding"] = data["Latent-Attention_Embedding"].apply(np.array)

    # Rename columns for consistency
    data.rename(columns={"Latent-Attention_Embedding": "Latent_Attention_Embedding"}, inplace=True)

    # Filter annotated rows
    data = data[data["annotated"] == True]

    # Prepare features and labels
    X = np.stack(data["Latent_Attention_Embedding"])
    y = data["Personal experience"].astype(int)

    return X, y


def train_and_evaluate_model(X, y, model_save_path):
    """
    Train a Random Forest model and evaluate its performance.

    Parameters:
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Target array.
    - model_save_path (str): Path to save the trained model.
    """
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=2000, n_jobs=-1, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Save the model
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_save_path}")

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Random Forest Classification Performance Metrics:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R-squared: {r2_score(y_test, y_pred):.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Generate and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # Plot predictions vs actuals
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], "--", color="red")
    plt.xlabel("Actual Phaticity Ratio")
    plt.ylabel("Predicted Phaticity Ratio")
    plt.title("Actual vs Predicted Phaticity Ratio")
    plt.show()

    # Return predictions and true labels
    return X_test, y_test, y_pred


if __name__ == "__main__":
    # Paths to data and model
    input_path = "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/output/embeddings/data_nv-embed_processed_output.pkl"
    model_save_path = "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/random_forest_models_smoteenn.pkl"

    # Load and preprocess data
    data = load_data(input_path)
    X, y = preprocess_data(data)

    # Train and evaluate model
    X_test, y_test, y_pred = train_and_evaluate_model(X, y, model_save_path)
