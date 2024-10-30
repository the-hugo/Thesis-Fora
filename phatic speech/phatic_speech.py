import pandas as pd

def load_data(input_path):
    print(f"Loading data from {input_path}")
    df = pd.read_pickle(input_path)
    return df

input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\data_llama70B_processed_output.pkl_temp_3200.pkl"

if __name__ == "__main__":
    print("Loading data")
    df = load_data(input_path)
    print("Data loaded")