import pandas as pd

# read in transformed_output.csv

df = pd.read_csv('../data/input/collection-24_UnitedWayDane_data.csv', sep=',')

# print the first 5 rows of the dataframe
print(df.columns)

