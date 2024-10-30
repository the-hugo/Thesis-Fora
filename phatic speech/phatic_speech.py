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
    
    # Adjust the pattern to include 0 or 1 as valid floats
    pattern = r"(.{0,40})(\d+\.\d+|0|1)"
    extracted_values = df["phatic speech"].str.extractall(pattern)

    # Combine preceding text and float value for context
    extracted_values["context_with_value"] = extracted_values[0] + extracted_values[1]

    # Filter for entries containing "Ratio" or "Calculation" in the preceding context
    filtered_values = extracted_values[extracted_values[0].str.contains(r"Ratio|Calculation", case=False, na=False)]

    # Select one match per row: prioritize "Ratio" over "Calculation"
    def select_preferred_match(group):
        # Check for presence of "Ratio" and prioritize it; otherwise, take the first match with "Calculation"
        ratio_match = group[group[0].str.contains("Ratio", case=False)]
        if not ratio_match.empty:
            return ratio_match.iloc[0]["context_with_value"]
        calculation_match = group[group[0].str.contains("Calculation", case=False)]
        if not calculation_match.empty:
            return calculation_match.iloc[0]["context_with_value"]
        return None

    # Apply the selection function to each row of the original index
    phaticity_ratio = filtered_values.groupby(level=0).apply(select_preferred_match)

    # Ensure alignment by reindexing to the original DataFrame
    df["phaticity ratio"] = phaticity_ratio.reindex(df.index).fillna("")

    print(df["phaticity score"])
    print("Phatic speech extracted")
    
