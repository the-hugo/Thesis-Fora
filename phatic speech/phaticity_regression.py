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
from sklearn.preprocessing import StandardScaler

# Load data from a pickle file
def load_data(input_path):
    """Load data from a pickle file."""
    print(f"Loading data from {input_path}")
    return pd.read_pickle(input_path)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == "__main__":
    input_path = "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/output/annotated/data_llama70B_processed_output_phatic_ratio.pkl_temp_24000_phatic_ratio.pkl"
    model_save_path = "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/random_forest_regressor.pkl"
    output_path = "C:/Users/paul-/Documents/Uni/Management and Digital Technologies/Thesis Fora/Code/data/output_filled_phatic_ratio.pkl"

    # Load and preprocess data
    data = load_data(input_path)
    data = data.dropna(subset=["Latent-Attention_Embedding"])
    data["Latent-Attention_Embedding"] = data["Latent-Attention_Embedding"].apply(np.array)
    data.rename(columns={"Latent-Attention_Embedding": "Latent_Attention_Embedding"}, inplace=True)

    # Separate rows with missing and non-missing phaticity ratio
    data_with_ratio = data.dropna(subset=["phaticity ratio"])
    data_missing_ratio = data[data["phaticity ratio"].isna()]

    # Define input (X) and target (y) for training the regressor
    X = np.hstack((np.stack(data_with_ratio["Latent_Attention_Embedding"].apply(lambda x: np.array(x))), data_with_ratio[["duration"]].values))
    y = data_with_ratio["phaticity ratio"].astype(float)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the regression model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_save_path}")

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    print("Random Forest Regression Performance Metrics:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared: {r2_score(y_test, y_pred)}")

    # Plotting predictions vs actuals
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], "--", color="red")
    plt.xlabel("Actual Phaticity Ratio")
    plt.ylabel("Predicted Phaticity Ratio")
    plt.title("Actual vs Predicted Phaticity Ratio")
    plt.show()

    # Predict missing phaticity ratio values
    if not data_missing_ratio.empty:
        X_missing = np.hstack((np.stack(data_missing_ratio["Latent_Attention_Embedding"].apply(lambda x: np.array(x))), data_missing_ratio[["duration"]].values))
        predicted_ratios = model.predict(X_missing)
        data.loc[data["phaticity ratio"].isna(), "phaticity ratio"] = predicted_ratios

    # Save the updated DataFrame with filled values
    data.to_pickle(output_path)
    print(f"Updated data saved to {output_path}")
