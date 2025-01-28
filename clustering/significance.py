import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact
import numpy as np

def analyze_clustering_effect(df):
    df = df[df['speaker_name'].duplicated(keep=False)]
    
    # Step 2: Identify speaker_name values that appear in more than one unique collection_id
    speaker_collection_counts = (
        df.groupby('speaker_name')['collection_id']
        .nunique()
        .reset_index(name='unique_collection_count')
    )
    # Keep only those speaker_names appearing in more than one collection_id
    valid_speakers = speaker_collection_counts[
        speaker_collection_counts['unique_collection_count'] > 1
    ]['speaker_name']
    
    # Step 3: Filter original DataFrame for these valid speakers
    df = df[df['speaker_name'].isin(valid_speakers)]
    
    contingency_table = pd.crosstab(df['speaker_name'], df['cluster'])
    print("\nContingency Table:")
    print(contingency_table)

    # Chi-Square Test of Independence
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    print("\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2_stat}, p-value: {p_value}, Degrees of Freedom: {dof}")
    print("Expected Frequencies:")
    print(expected)

    # Calculate Cramér's V
    n = contingency_table.to_numpy().sum()  # Total observations
    k = min(contingency_table.shape)  # Smaller dimension (rows or columns)
    cramers_v = np.sqrt(chi2_stat / (n * (k - 1)))
    print(f"\nCramér's V: {cramers_v:.4f}")
    
    # Interpretation of Cramér's V
    if cramers_v < 0.1:
        effect_size = "weak"
    elif cramers_v < 0.3:
        effect_size = "small"
    elif cramers_v < 0.5:
        effect_size = "moderate"
    else:
        effect_size = "strong"
    print(f"Effect Size (Cramér's V): {effect_size}")

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues")
    plt.title("Heatmap of Collection IDs vs Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Collection ID")
    plt.show()

    # Optional: Fisher Exact Test (if 2x2 table)
    if contingency_table.shape == (2, 2):
        oddsratio, fisher_p = fisher_exact(contingency_table)
        print("\nFisher's Exact Test Results:")
        print(f"Odds Ratio: {oddsratio}, p-value: {fisher_p}")

# Assuming your DataFrame has columns 'conversation_id', 'cluster', and 'collection_id'
if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\facilitators_features_clustered.csv"
    df = pd.read_csv(input_path)

    # Ensure the DataFrame contains the necessary columns
    if {'conversation_id', 'cluster', 'collection_id'}.issubset(df.columns):
        analyze_clustering_effect(df)
    else:
        print("DataFrame does not contain required columns: 'conversation_id', 'cluster', 'collection_id'.")
