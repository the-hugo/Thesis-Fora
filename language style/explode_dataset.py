import spacy
import pandas as pd

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to tokenize text into sentences
def tokenize_to_sentences(text):
    """
    Tokenize text into sentences using SpaCy.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

# Function to group sentences into chunks
def group_sentences(sentences, max_words=100):
    """
    Group sentences into chunks without exceeding the max word limit.
    """
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        # Check if adding the current sentence exceeds the max word limit
        if current_word_count + word_count <= max_words:
            current_chunk.append(sentence)
            current_word_count += word_count
        else:
            # Add the current chunk to chunks
            chunks.append(" ".join(current_chunk))
            # Start a new chunk with the current sentence
            current_chunk = [sentence]
            current_word_count = word_count

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to process a single text snippet
def process_text_snippet(text, max_words=100):
    """
    Process a single text snippet by splitting into chunks based on max word limit.
    """
    sentences = tokenize_to_sentences(text)
    return group_sentences(sentences, max_words)

# Function to process the DataFrame and split rows based on chunks
def process_dataframe(df, text_column, max_words=100):
    """
    Process the DataFrame, splitting text snippets into chunks and replicating rows.
    """
    processed_rows = []

    for _, row in df.iterrows():
        try:
            text = row[text_column]
            chunks = process_text_snippet(text, max_words)
            for chunk in chunks:
                new_row = row.copy()
                new_row[text_column] = chunk
                processed_rows.append(new_row)
        except ValueError:
            print(text)

    # Create a new DataFrame with the processed rows
    return pd.DataFrame(processed_rows)

# Function to load data from a CSV file
def load_data(input_path):
    """
    Load data from a CSV file.
    """
    print(f"Loading data from {input_path}")
    return pd.read_pickle(input_path)

# Main function
if __name__ == "__main__":
    # Load the data
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\data_llama70B_processed_output.pkl"
    df = load_data(input_path)

    # Column containing the text snippets
    text_column = "words"
    max_words = 60  # Adjust word limit per chunk

    # Process the DataFrame
    processed_df = process_dataframe(df, text_column, max_words)

    # Save the processed DataFrame to a new file
    output_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\processed_chunks.csv"
    processed_df.to_csv(output_path, index=False)

    print(f"Processed data saved to {output_path}")
