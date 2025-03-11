import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon

def load_data(input_path):
    print(f"Loading data from {input_path}")
    df = pd.read_pickle(input_path)
    return df

def test_phaticity_diff(df):
    # Perform paired t-test
    t_stat, t_pvalue = ttest_rel(df['mean_conversation_phaticity_ratio_participants'],
                                 df['mean_conversation_phaticity_ratio_fac'])
    print(f"Paired t-test results: t-statistic = {t_stat}, p-value = {t_pvalue}")

    # Perform Wilcoxon signed-rank test
    w_stat, w_pvalue = wilcoxon(df['mean_conversation_phaticity_ratio_participants'],
                                df['mean_conversation_phaticity_ratio_fac'])
    print(f"Wilcoxon signed-rank test results: W-statistic = {w_stat}, p-value = {w_pvalue}")

def regression_with_speaker_count(df):
    # Calculate phaticity_diff as the difference between facilitator and participant ratios
    df['phaticity_diff'] = df['mean_conversation_phaticity_ratio_fac'] - df['mean_conversation_phaticity_ratio_participants']
    
    # Fit a linear regression model with phaticity_diff as the dependent variable and speaker_count as the independent variable
    X = df[['mean_spkr_count']]
    X = sm.add_constant(X)  # Adds the intercept
    y = df['phaticity_diff']
    
    model = sm.OLS(y, X).fit()
    print("\nRegression results:")
    print(model.summary())

    # Calculate the correlation between phaticity_diff and speaker_count
    correlation = df['phaticity_diff'].corr(df['mean_spkr_count'])
    print(f"\nCorrelation between phaticity_diff and speaker_count: {correlation}")

    # Plotting
    # 1. Box Plot for phaticity_diff across speaker_count quartiles
    df['speaker_count_quartile'] = pd.qcut(df['mean_spkr_count'], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='speaker_count_quartile', y='phaticity_diff', data=df)
    plt.title("Box Plot of phaticity_diff across speaker_count Quartiles")
    plt.xlabel("Speaker Count Quartile")
    plt.ylabel("Phaticity Difference (Facilitator - Participant)")
    plt.show()

    # 2. Scatter Plot with Regression Line
    plt.figure(figsize=(10, 6))
    sns.regplot(x='mean_spkr_count', y='phaticity_diff', data=df, ci=None, line_kws={"color": "red"})
    plt.title("Scatter Plot of phaticity_diff vs. Speaker Count with Regression Line")
    plt.xlabel("Mean Speaker Count")
    plt.ylabel("Phaticity Difference (Facilitator - Participant)")
    plt.show()

    # 3. Paired Bar Plot of phaticity ratios
    plt.figure(figsize=(10, 6))
    df_melted = df.melt(id_vars=['conversation_id'], 
                        value_vars=['mean_conversation_phaticity_ratio_participants', 'mean_conversation_phaticity_ratio_fac'], 
                        var_name='Phaticity Type', value_name='Phaticity Ratio')
    sns.barplot(x='conversation_id', y='Phaticity Ratio', hue='Phaticity Type', data=df_melted)
    plt.title("Comparison of Phaticity Ratios for Facilitators and Participants")
    plt.xticks([], [])  # Hide x-axis labels for readability
    plt.ylabel("Mean Phaticity Ratio")
    plt.legend(title='Phaticity Type', labels=['Participants', 'Facilitators'])
    plt.show()

if __name__ == "__main__":
    input_path = r"C:\Users\paul-\Documents\Uni\Management and Digital Technologies\Thesis Fora\Code\data\output\annotated\output_filled_phatic_ratio_conversation_speaker_phatic_ratio.pkl"
    print("Loading data")
    df = load_data(input_path)
    # please give me max and min duration and mean duration
    print(f"Max duration: {df['duration'].max()}")
    print(f"Min duration: {df['duration'].min()}")
    print(f"Mean duration: {df['duration'].mean()}")
    print("Data loaded")
    
    # Group by conversation_id and calculate the mean of the ratios and speaker_count
    df = df.groupby("conversation_id").agg(
        mean_conversation_phaticity_ratio_fac=("conversation_phaticity_ratio_fac", "mean"),
        mean_conversation_phaticity_ratio_participants=("conversation_phaticity_ratio_participants", "mean"),
        mean_spkr_count=("speaker_count", "mean")
    ).reset_index()

    # Run the test to see if there is a significant difference in phaticity ratios
    test_phaticity_diff(df)

    # Run regression and correlation analysis to check the effect of speaker_count on phaticity_diff
    regression_with_speaker_count(df)
