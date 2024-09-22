import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data(input_path):
    """Load data from a pickle file."""
    print(f"Loading data from {input_path}")
    return pd.read_pickle(input_path)

def split_data(df):
    """Split the data into annotated and unannotated subsets."""
    df_annotated = df[df['annotated']].copy()
    df_unannotated = df[~df['annotated']].copy()
    return df_annotated, df_unannotated

def preprocess_data(df):
    """Preprocess the data by converting columns to numeric and handling embeddings."""
    df['SpeakerTurn'] = pd.to_numeric(df['SpeakerTurn'])
    df['audio_start_offset'] = pd.to_numeric(df['audio_start_offset'])
    df['audio_end_offset'] = pd.to_numeric(df['audio_end_offset'])
    df['is_fac'] = pd.to_numeric(df['is_fac']).astype(int)
    df['cofacilitated'] = pd.to_numeric(df['cofacilitated']).astype(int)
    df['Latent-Attention_Embedding'] = df['Latent-Attention_Embedding'].apply(np.array)
    return df

def undersample_data(X, y):
    """Perform random undersampling of the majority class."""
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res

def split_train_test(df, predictors, target, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    X = df[predictors]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Perform undersampling on the training data
    X_train_res, y_train_res = undersample_data(X_train, y_train)
    
    return X_train_res, y_train_res, X_test, y_test

def df_to_tensors(X, y, target_embedding_length):
    """Convert the dataframe columns to tensors for input to the neural network."""
    embeddings = np.vstack(
        X['Latent-Attention_Embedding'].apply(
            lambda x: np.pad(x, (0, max(0, target_embedding_length - len(x))))[:target_embedding_length]
        ).values
    )
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)

    other_features = torch.tensor(X[['SpeakerTurn', 'audio_start_offset', 'audio_end_offset', 'is_fac', 'cofacilitated']].values, dtype=torch.float32).to(device)

    X_tensor = torch.cat([other_features, embeddings_tensor], dim=1)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)

    return X_tensor, y_tensor

def compute_class_weights(y_train):
    """Compute class weights to handle imbalanced datasets."""
    class_weights = {}
    for column in y_train.columns:
        pos_count = y_train[column].sum()
        neg_count = len(y_train) - pos_count
        class_weights[column] = neg_count / (pos_count + 1e-5)

    pos_weights = torch.tensor([class_weights[col] for col in y_train.columns], dtype=torch.float32).to(device)
    
    return pos_weights

class ResidualBlock(nn.Module):
    """Residual Block with two linear layers."""
    def __init__(self, input_size, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.bn2 = nn.BatchNorm1d(input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += identity  # Residual connection
        return self.relu(out)

class ComplexNN(nn.Module):
    """More complex neural network with convolutional layers and residual connections."""
    def __init__(self, input_size, output_size):
        super(ComplexNN, self).__init__()
        
        # Convolutional layers for processing embeddings
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(input_size, 512)
        self.residual_block = ResidualBlock(512, 256)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Assuming the embedding is reshaped for 1D convolution
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Flatten the convolution output for fully connected layers
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(self.residual_block(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def evaluate_model(y_true, y_pred, target):
    """Evaluate model performance using accuracy, precision, recall, f1-score, AUC-ROC, and confusion matrix."""
    y_pred_prob = torch.sigmoid(y_pred).cpu().numpy()
    y_pred_bin = (y_pred_prob > 0.5).astype(int)
    y_true_np = y_true.cpu().numpy()

    accuracy = accuracy_score(y_true_np, y_pred_bin)
    precision = precision_score(y_true_np, y_pred_bin, average='macro')
    recall = recall_score(y_true_np, y_pred_bin, average='macro')
    f1 = f1_score(y_true_np, y_pred_bin, average='macro')
    auc_roc = roc_auc_score(y_true_np, y_pred_prob, average='macro')

    for i, label in enumerate(target):
        print(f"\nConfusion matrix for {label}:")
        cm = confusion_matrix(y_true_np[:, i], y_pred_bin[:, i])
        plot_confusion_matrix(cm, label)

    return accuracy, precision, recall, f1, auc_roc

def plot_confusion_matrix(cm, label):
    """Plot the confusion matrix."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {label}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def main(input_path, model_save_path, num_epochs=50, batch_size=32, learning_rate=0.0001):
    """Main function to load data, preprocess, train and evaluate the model."""
    df = load_data(input_path)
    
    df_annotated, df_unannotated = split_data(df)
    df_annotated = preprocess_data(df_annotated)
    
    predictors = ['SpeakerTurn', 'audio_start_offset', 'audio_end_offset', 'is_fac', 'cofacilitated', 'Latent-Attention_Embedding']
    target = ['Personal story', 'Personal experience', 'Express affirmation', 'Specific invitation', 
              'Provide example', 'Open invitation', 'Make connections', 'Express appreciation', 'Follow up question']
    
    X_train, y_train, X_test, y_test = split_train_test(df_annotated, predictors, target)
    pos_weights = compute_class_weights(y_train)
    target_embedding_length = 4096

    # Prepare data tensors
    X_train_tensor, y_train_tensor = df_to_tensors(X_train, y_train, target_embedding_length)
    X_test_tensor, y_test_tensor = df_to_tensors(X_test, y_test, target_embedding_length)

    input_size = target_embedding_length + 5

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = ComplexNN(input_size=input_size, output_size=len(target)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Training loop with learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}")

    # Evaluate the model on the test set
    model.eval()
    test_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            all_outputs.append(outputs)
            all_targets.append(batch_y)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    accuracy, precision, recall, f1, auc_roc = evaluate_model(all_targets, all_outputs, target)

    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\data_nv-embed_processed_output.pkl"
    model_save_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\complex_nn_model_classification.pth"
    main(input_path, model_save_path, num_epochs=50, batch_size=32, learning_rate=0.0001)
