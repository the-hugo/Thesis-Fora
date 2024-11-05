import pandas as pd
import statsmodels.formula.api as smf

def load_data(input_path):
    print(f"Loading data from {input_path}")
    df = pd.read_pickle(input_path)
    return df

def check_speaker_count_effect(df):
    # Run a regression with phaticity_diff as the dependent variable and speaker_count as the independent variable
    model = smf.ols('phaticity_diff ~ speaker_count', data=df).fit()
    
    # Print the regression summary to check the effect of speaker_count on phaticity_diff
    print(model.summary())

if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\output_filled_phatic_ratio_conversation_speaker_phatic_ratio.pkl"
    print("Loading data")
    df = load_data(input_path)
    print("Data loaded")

    # Check if speaker_count affects phaticity_diff
    check_speaker_count_effect(df)
