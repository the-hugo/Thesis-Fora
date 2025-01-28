import pandas as pd


def redact_facilitators(df):
    df.loc[(df["speaker_id"] == 37198) & (df["conversation_id"] == 2062), "is_fac"] = True
    df.loc[(df["speaker_id"] == 28749) & (df["conversation_id"] == 719), "is_fac"] = True
    df.loc[(df["speaker_name"] == "La [Surname]") & (df["conversation_id"] == 2294), "is_fac"] = False
    df.loc[(df["speaker_name"] == "Mathias") & (df["conversation_id"] == 916), "is_fac"] = False
    df.loc[(df["speaker_name"] == "Mathias") & (df["conversation_id"] == 914), "is_fac"] = False
    df.loc[(df["speaker_name"] == "Keylynne") & (df["conversation_id"] == 831), "is_fac"] = False
    df.loc[(df["speaker_name"] == "Mathias") & (df["conversation_id"] == 804), "is_fac"] = False
    df.loc[(df["speaker_name"] == "Mathias") & (df["conversation_id"] == 785), "is_fac"] = False
    df.loc[(df["speaker_name"] == "Beatriz") & (df["conversation_id"] == 717), "is_fac"] = False
    df.loc[(df["speaker_name"] == "Interpreter") & (df["conversation_id"] == 717), "speaker_name"] = "Kara Guzman"
    df.loc[(df["speaker_name"] == "Mathias") & (df["conversation_id"] == 674), "is_fac"] = False
    return df


data = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\embeddings\data_nv-embed_processed_output.pkl"
df = pd.read_pickle(data)
df = redact_facilitators(df)
# save the redacted data
output_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\embeddings\data_nv-embed_processed_output.pkl"
df.to_pickle(output_path)