import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import umap

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simplified data loading function
def load_data(input_path):
    return pd.read_pickle(input_path)


def preprocess_data(df, n_components=256, target_length=4096):
    """Preprocess the data by handling inconsistent embedding sizes and applying UMAP."""
    
    # Ensure embeddings have the same size (pad or truncate to target_length)
    def pad_or_truncate(embedding, target_length):
        # Handle None or NaN embeddings
        if embedding is None or not isinstance(embedding, (list, np.ndarray)):
            embedding = np.zeros(target_length)
        
        embedding = np.array(embedding)  # Ensure it's a NumPy array
        
        # Pad or truncate to the target length
        if len(embedding) < target_length:
            return np.pad(embedding, (0, target_length - len(embedding)), 'constant')
        else:
            return embedding[:target_length]
    
    # Apply the function to standardize all embeddings
    df['Latent-Attention_Embedding'] = df['Latent-Attention_Embedding'].apply(lambda x: pad_or_truncate(x, target_length))
    
    # Stack the embeddings into a matrix
    embeddings = np.vstack(df['Latent-Attention_Embedding'].values)
    
    # Apply UMAP for dimensionality reduction
    umap_model = umap.UMAP(n_components=n_components, random_state=42)
    embeddings_reduced = umap_model.fit_transform(embeddings)
    
    # Update the dataframe with reduced embeddings
    df['Latent-Attention_Embedding'] = list(embeddings_reduced)
    
    return df


# Train-test split with undersampling
def split_train_test(df, predictors, target, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[target], test_size=test_size)
    X_train_res, y_train_res = RandomUnderSampler().fit_resample(X_train, y_train)
    return X_train_res, y_train_res, X_test, y_test

# Convert DataFrame to Tensors
def df_to_tensors(X, y, embedding_length):
    embeddings = torch.tensor(np.vstack(X['Latent-Attention_Embedding']), dtype=torch.float32).to(device)
    features = torch.tensor(X.drop(columns=['Latent-Attention_Embedding']).values, dtype=torch.float32).to(device)
    X_tensor = torch.cat([features, embeddings], dim=1)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)
    return X_tensor, y_tensor

# Simple transformer model
class TransformerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Main function
def main(input_path, model_save_path, num_epochs=10, batch_size=32, lr=0.001):
    df = preprocess_data(load_data(input_path))
    predictors = ['SpeakerTurn', 'audio_start_offset', 'audio_end_offset', 'is_fac', 'cofacilitated', 'Latent-Attention_Embedding']
    target = 'Personal story'  # Change this to your target labels loop if needed
    
    X_train, y_train, X_test, y_test = split_train_test(df, predictors, target)
    X_train_tensor, y_train_tensor = df_to_tensors(X_train, y_train, 256)
    X_test_tensor, y_test_tensor = df_to_tensors(X_test, y_test, 256)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)

    model = TransformerNN(input_size=261, output_size=1).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\data_nv-embed_processed_output.pkl"
    main(input_path, model_save_path="model.pth")
