import pandas as pd


def load_data(input_path):
    print(f"Loading data from {input_path}")
    df = pd.read_pickle(input_path)
    return df


if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\output_filled_phatic_ratio.pkl"
    print("Loading data")
    df = load_data(input_path)
    print("Data loaded")
    df["speaker_name"] = df["speaker_name"].str.lower()
    df = df[
        ~df["speaker_name"].str.contains(
            "^speaker|moderator|audio|computer|computer voice|facilitator|group|highlight|interpreter|interviewer|multiple voices|other speaker|participant|redacted|speaker X|unknown|video"
        )
    ]

    # merge speaker names that are the same except [Surname] string within the same collection_id
    df["speaker_name"] = df["speaker_name"].str.replace(r"\[.*\]", "", regex=True)
    df["speaker_name"] = df["speaker_name"].str.strip()
    df["phaticity ratio"] = df["phaticity ratio"].apply(lambda x: 1 if x >= 0.5 else 0)

    # Calculate phaticity ratio by conversation and speaker
    conversation_speaker_turns = (
        df.groupby(["conversation_id", "speaker_name", "is_fac", "speaker_id"])
        .agg(
            phaticity_ratio_1_count=("phaticity ratio", lambda x: (x == 1).sum()),
            phaticity_ratio_all_count=("phaticity ratio", "count"),
            personal_experience_count=("Personal experience", "sum"),
            personal_story_count=("Personal story", "sum"),
        )
        .reset_index()
    )
        
    conversation_speaker_turns["conversation_speaker_phaticity_ratio"] = (
        conversation_speaker_turns["phaticity_ratio_1_count"]
        / conversation_speaker_turns["phaticity_ratio_all_count"]
    )
    
    conversation_speaker_turns = conversation_speaker_turns[conversation_speaker_turns["phaticity_ratio_all_count"] > 3]
    
    # for every conversation calculate the average phaticity ratio by sum of phaticity ratio 1 count and phaticity ratio all count
    # Calculate conversation phaticity ratio for facilitators
    fac_turns = conversation_speaker_turns[conversation_speaker_turns["is_fac"] == 1]
    fac_turns["conversation_phaticity_ratio_fac"] = (
        fac_turns.groupby(["conversation_id"])["phaticity_ratio_1_count"]
        .transform("sum")
        / fac_turns.groupby(["conversation_id"])["phaticity_ratio_all_count"]
        .transform("sum")
    )
    
    # Calculate conversation phaticity ratio for participants
    part_turns = conversation_speaker_turns[conversation_speaker_turns["is_fac"] == 0]
    part_turns["conversation_phaticity_ratio_participants"] = (
        part_turns.groupby(["conversation_id"])["phaticity_ratio_1_count"]
        .transform("sum")
        / part_turns.groupby(["conversation_id"])["phaticity_ratio_all_count"]
        .transform("sum")
    )
    
    # Merge the results back into the original dataframe
    conversation_speaker_turns = conversation_speaker_turns.merge(
        fac_turns[["conversation_id", "speaker_name", "conversation_phaticity_ratio_fac"]],
        on=["conversation_id", "speaker_name"],
        how="left"
    )
    
    conversation_speaker_turns = conversation_speaker_turns.merge(
        part_turns[["conversation_id", "speaker_name", "conversation_phaticity_ratio_participants"]],
        on=["conversation_id", "speaker_name"],
        how="left"
    )
    
    # calculate the overall phaticity ratio by grouping by conversation_id
    conversation_speaker_turns["conversation_phaticity_ratio"] = (
        conversation_speaker_turns.groupby(["conversation_id"])["phaticity_ratio_1_count"]
        .transform("sum")
        / conversation_speaker_turns.groupby(["conversation_id"])["phaticity_ratio_all_count"]
        .transform("sum")
    )
    
    # for every conversation, fillnas of conversation_phaticity_ratio_fac and conversation_phaticity_ratio_participants
    conversation_speaker_turns["conversation_phaticity_ratio_fac"] = conversation_speaker_turns.groupby("conversation_id")["conversation_phaticity_ratio_fac"].transform("mean")
    conversation_speaker_turns["conversation_phaticity_ratio_participants"] = conversation_speaker_turns.groupby("conversation_id")["conversation_phaticity_ratio_participants"].transform("mean")
 
    # calculate the difference between the fac phaticity and participant phaticity
    conversation_speaker_turns["phaticity_diff"] = (
        conversation_speaker_turns["conversation_phaticity_ratio_fac"]
        - conversation_speaker_turns["conversation_phaticity_ratio_participants"]
    )
    
    # Round final results to 2 decimals
    conversation_speaker_turns["conversation_phaticity_ratio_fac"] = conversation_speaker_turns["conversation_phaticity_ratio_fac"].round(2)
    conversation_speaker_turns["conversation_phaticity_ratio_participants"] = conversation_speaker_turns["conversation_phaticity_ratio_participants"].round(2)
    conversation_speaker_turns["conversation_phaticity_ratio"] = conversation_speaker_turns["conversation_phaticity_ratio"].round(2)
    conversation_speaker_turns["phaticity_diff"] = conversation_speaker_turns["phaticity_diff"].round(2)
    conversation_speaker_turns["conversation_speaker_phaticity_ratio"] = conversation_speaker_turns["conversation_speaker_phaticity_ratio"].round(2)
    # eliminate rows where conversation_phaticity_ratio_fac is null
    conversation_speaker_turns = conversation_speaker_turns[~conversation_speaker_turns["conversation_phaticity_ratio_fac"].isna()]
    
    # add a column that counts the total speaker count per conversation
    conversation_speaker_turns["speaker_count"] = conversation_speaker_turns.groupby("conversation_id")["speaker_name"].transform("count")
    
    # merge with the original dataframe
    conversation_speaker_turns.to_pickle(input_path.replace(".pkl", "_conversation_speaker_phatic_ratio.pkl"))
    conversation_speaker_turns.to_csv(input_path.replace(".pkl", "_conversation_speaker_phatic_ratio.csv"))
    print("done")
