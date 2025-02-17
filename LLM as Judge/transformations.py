import pandas as pd

def main():
    # Load the pickle file (update 'data.pkl' to your file path if needed)
    df = pd.read_pickle(r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\data\speech_acts_personal_sharing_classification.pkl")
    
    # --- Transformation 1: Update Title-Case Columns Based on Lower-Case Columns ---
    # Define mapping from lower-case column names to the corresponding title-case column names.
    col_pairs = {
        'express_appreciation': 'Express appreciation',
        'express_affirmation': 'Express affirmation',
        'open_invitation': 'Open invitation',
        'specific_invitation': 'Specific invitation'
    }
    
    # For each pair, update the title-case column to 1 if the lower-case column is 1 and the title-case column is 0.
    for lower_col, upper_col in col_pairs.items():
        if lower_col in df.columns and upper_col in df.columns:
            condition = (df[lower_col] == 1) & (df[upper_col] == 0)
            df.loc[condition, upper_col] = 1
    
    # Drop the lower-case columns after processing.
    df.drop(columns=list(col_pairs.keys()), inplace=True, errors='ignore')
    df.drop(columns=["speech_acts_result"], inplace=True, errors='ignore')
    # --- Transformation 2: Update 'Personal experience' and 'Personal story' Based on personal_sharing_result ---
    if 'personal_sharing_result' in df.columns:
        # Convert to string in case there are missing or non-string values.
        sharing_series = df['personal_sharing_result'].astype(str)
        
        # If the existing 'Personal experience' column is present, update it to 1 where applicable.
        if 'Personal experience' in df.columns:
            experience_condition = sharing_series.str.contains('Personal experience', na=False) & (df['Personal experience'] == 0)
            df.loc[experience_condition, 'Personal experience'] = 1
        
        # Similarly, update 'Personal story' column.
        if 'Personal story' in df.columns:
            story_condition = sharing_series.str.contains('Personal story', na=False) & (df['Personal story'] == 0)
            df.loc[story_condition, 'Personal story'] = 1
        
        # Drop the personal_sharing_result column after processing.
        df.drop(columns=['personal_sharing_result'], inplace=True)
    
    # Save the transformed DataFrame to a new pickle file.
    df.to_pickle('transformed_data.pkl')
    df.to_csv('transformed_data.csv', index=False)
    print("Transformation complete. The updated DataFrame has been saved as 'transformed_data.pkl'.")

if __name__ == '__main__':
    main()
