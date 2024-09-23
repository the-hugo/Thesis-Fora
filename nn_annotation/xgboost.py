import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import umap

# Load the data
input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\data_nv-embed_processed_output.pkl"
print("Loading data from:", input_path)
df = pd.read_pickle(input_path)
print("Data loaded successfully.")

print("Filtering rows with non-null 'Latent-Attention_Embedding'...")
df = df[df['Latent-Attention_Embedding'].notnull()]
print(f"Rows remaining after filtering non-null embeddings: {len(df)}")

print("Checking embedding length and filtering rows with incorrect length...")
df['embedding_length'] = df['Latent-Attention_Embedding'].apply(lambda x: len(x) if x is not None else 0)
initial_count = len(df)
df_filtered = df[df['embedding_length'] == 4096]
filtered_count = len(df_filtered)
print(f"Number of rows filtered out: {initial_count - filtered_count}")
df = df_filtered
print(f"Rows remaining after filtering by embedding length: {len(df)}")

print("Extracting embeddings and target labels...")
X = np.vstack(df['Latent-Attention_Embedding'].values)
y = df['Personal story']  # Assuming 'Target' is the column with your target labels
print("Embeddings and target labels extracted.")

print("Applying UMAP for dimensionality reduction...")
umap_model = umap.UMAP(n_components=50, random_state=42)  # You can adjust n_components
X_umap = umap_model.fit_transform(X)
print("UMAP dimensionality reduction completed.")

print("Encoding target labels to integers...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Target labels encoded.")

print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_umap, y_encoded, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

print("Training the SVM classifier...")
svm_model = SVC(kernel='linear', probability=True)  # You can try 'rbf' or other kernels as well
svm_model.fit(X_train, y_train)
print("SVM classifier trained.")

print("Predicting on the test set...")
y_pred = svm_model.predict(X_test)
print("Prediction completed.")

print("Evaluating the model...")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Model evaluation completed.")
