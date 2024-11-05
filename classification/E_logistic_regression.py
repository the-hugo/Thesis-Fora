import pandas as pd
import umap
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


# Load data from a pickle file
def load_data(input_path):
    """Load data from a pickle file."""
    print(f"Loading data from {input_path}")
    return pd.read_pickle(input_path)


if __name__ == "__main__":
    input_path = "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/output/embeddings/data_nv-embed_processed_output.pkl"
    model_save_path = "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/random_forest_models_smoteenn.pkl"

    data = load_data(input_path)
    
    data = data.dropna(subset=["Latent-Attention_Embedding"])

    data["Latent-Attention_Embedding"] = data["Latent-Attention_Embedding"].apply(
        np.array
    )

    data.rename(
        columns={"Latent-Attention_Embedding": "Latent_Attention_Embedding"},
        inplace=True,
    )

    conversation_info_columns = [
        # "collection_title",
        # "symbol",
        "is_fac"
        # "conversation_ids",
        # "speaker_id",
        # "speaker_name",
    ]
    conversation_info = (
        data.groupby(["conversation_id", "speaker_name"])
        .apply(
            lambda x: {
                # "collection_title": x["collection_title"].unique()[0],
                # "symbol": x["symbol"].unique()[0],
                "is_fac": x["is_fac"].unique()[0],
                # "conversation_ids": ", ".join(map(str, x["conversation_id"].unique())),
                # "speaker_id": ", ".join(map(str, x["speaker_id"].unique())),
                # "speaker_name": ", ".join(map(str, x["speaker_name"].unique())),
            }
        )
        .reset_index()
    )
    
    """
    data = (
        data.groupby(["conversation_id", "speaker_name"])
        .agg(
            Latent_Attention_Embedding=(
                "Latent_Attention_Embedding",
                lambda x: np.mean(x, axis=0),
            ),
        )
        .reset_index()
    )
    conversation_info[conversation_info_columns] = pd.DataFrame(
        conversation_info[0].tolist(), index=conversation_info.index
    )
    data = pd.merge(data, conversation_info, on=["conversation_id", "speaker_name"])
    """
    
    X = np.stack(
        data["Latent_Attention_Embedding"].apply(lambda x: np.array(x))
    )

    y = data["is_fac"].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)

    with open(model_save_path, "wb") as f:
        pickle.dump(logistic_model, f)
    print(f"Model saved to {model_save_path}")

    y_pred = logistic_model.predict(X_test)

    print("Logistic Regression Performance Metrics:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=logistic_model.classes_
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
