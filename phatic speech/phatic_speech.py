import pandas as pd
import re
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

def load_data(input_path):
    print(f"Loading data from {input_path}")
    df = pd.read_pickle(input_path)
    return df


def match_pattern(df):
    # Adjust the pattern to include 0 or 1 as valid floats
    pattern = re.compile(r"(.{0,40})(\d+\.\d+|0|1)(.{0,10})", re.DOTALL)
    extracted_values = df["phatic speech"].str.extractall(pattern)

    # Combine preceding text and float value for context
    extracted_values["context_with_value"] = (
        extracted_values[0] + extracted_values[1] + extracted_values[2]
    )

    # Filter for entries containing "Ratio" or "Calculation" in the preceding context
    filtered_values = extracted_values[
        extracted_values[0].str.contains(r"Ratio|Calculation", case=False, na=False)
    ]

    # Select one match per row: prioritize "Ratio" over "Calculation"
    def select_preferred_match(group):
        # Check for presence of "Ratio Calculation" and prioritize it; otherwise, take the first match with "Calculation"
        ratio_match = group[group[0].str.contains(r"\bRatio Calculation\b", case=False)]
        if not ratio_match.empty:
            return ratio_match.iloc[0]["context_with_value"]
        calculation_match = group[group[0].str.contains("Phatic Ratio", case=False)]
        if not calculation_match.empty:
            return calculation_match.iloc[0]["context_with_value"]
        return None

    # Apply the selection function to each row of the original index
    phaticity_ratio = filtered_values.groupby(level=0).apply(select_preferred_match)

    # Ensure alignment by reindexing to the original DataFrame
    df["phaticity ratio"] = phaticity_ratio.reindex(df.index).fillna("")

    # in phaticity ratio kill everything before Phatic, if there is no "Phatic" kill everything before Ratio
    df["phaticity ratio"] = df["phaticity ratio"].str.replace(
        r".*(Phatic\s*Ratio|Phatic|Ratio)", r"\1", case=False, regex=True
    )
    # extract this pattern: [number or float] / [number or float]
    pattern = re.compile(r"(\d+\.\d+|\d+)\s*/\s*(\d+\.\d+|\d+)", re.DOTALL)
    extracted_values = df["phaticity ratio"].str.extract(pattern)
    # combine the extracted values
    df["phaticity ratio"] = (
        extracted_values[0].astype(float) / extracted_values[1].astype(float)
    ).round(2)

    print("Done")
    return df


def plot_histogram(df):
    # Calculate conversation duration and normalized time
    df["conversation_duration"] = df.groupby("conversation_id")[
        "audio_end_offset"
    ].transform("max") - df.groupby("conversation_id")["audio_start_offset"].transform(
        "min"
    )
    df["normalized_time"] = (
        100
        * (
            df["audio_start_offset"]
            - df.groupby("conversation_id")["audio_start_offset"].transform("min")
        )
        / df["conversation_duration"]
    )
    
    # Discretize normalized time into intervals of 1%
    df["normalized_time"] = df["normalized_time"].round().astype(int)
    
    # Calculate the sum of phaticity ratio and turn count per normalized time interval
    phaticity_df = df.groupby("normalized_time").agg(
        phaticity_ratio_sum=("phaticity ratio", "sum"),
        turn_count=("conversation_id", "count")
    ).reset_index()
    phaticity_df["average_phaticity_ratio"] = phaticity_df["phaticity_ratio_sum"] / phaticity_df["turn_count"]

    # Plot histogram using Plotly
    fig = px.bar(
        phaticity_df, 
        x="normalized_time", 
        y="average_phaticity_ratio",
        labels={"normalized_time": "Normalized Time (%)", "average_phaticity_ratio": "Average Phaticity Ratio"},
        title="Average Phaticity Ratio Distribution Over Normalized Time",
    )
    fig.update_layout(bargap=0.2)
    fig.show()


if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\data_llama70B_processed_output.pkl_temp_7200.pkl"
    print("Loading data")
    df = load_data(input_path)
    print("Data loaded")

    print("Matching pattern")
    df = match_pattern(df)
    print("Pattern matched")

    print("Plotting histogram")
    conversation = False
    # plot the phaticity ratio over normalised time
    plot_histogram(df)
    
    # save as pickle and csv
    df.to_pickle(input_path.replace(".pkl", "_phatic_ratio.pkl"))
    df.to_csv(input_path.replace(".pkl", "_phatic_ratio.csv"))
