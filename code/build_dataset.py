"""
This file builds the dataset used by my project.
"""
from datetime import datetime
import pandas as pd


# read in evictions and unemployment
path_to_cleaned_evictions_data = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/cleaned_data/evictions.csv"
path_to_cleaned_unemployment_data = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/cleaned_data/unemployment.csv"
path_to_cleaned_oi_data = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/cleaned_data/covariates.csv"
path_to_final_dataset = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/cleaned_data/cleaned_dataset.csv"

evictions_df = pd.read_csv(path_to_cleaned_evictions_data)
unemployment_df = pd.read_csv(path_to_cleaned_unemployment_data)

# join on tract, month
print(evictions_df.columns)
print(unemployment_df.columns)
evictions_and_unemployment = evictions_df.merge(unemployment_df, how='inner', on=['tract', 'month'])

# pandas saves index columns from each individual dataset; drop them
evictions_and_unemployment = evictions_and_unemployment.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'])

# now we merge with covariates
oi_df = pd.read_csv("/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/cleaned_data/covariates.csv")
merged_df = evictions_and_unemployment.merge(oi_df,
                                             how='inner',
                                             on='tract')

# pandas saves index columns from each individual dataset; drop them
merged_df = merged_df.drop(columns='Unnamed: 0')

# created lagged columns and drop non-lagged columns, to reflect information available to policymakers at prediction time
cols_to_lag = ['cnip', 'unemployment_rate', 'filings']
for lag in range(1, 2):
    for col in cols_to_lag:
        lagged_col_name = "L" + str(lag) + "_" + col
        merged_df[lagged_col_name] = merged_df.groupby('tract').shift(lag)[col]
merged_df = merged_df.drop(columns=['cnip', 'unemployment_rate'])  # drop current month CNIP and unemployment rate

# keep only observations which have non-missing values for both lags (obs. from March-August inclusive)
merged_df = merged_df.drop(merged_df.loc[pd.to_datetime(merged_df['month']) < datetime(2020, 2, 1)].index)

# save FIPS codes as strings so that their column titles are strings when we generate dummy variables later
merged_df['tract'] = merged_df['tract'].astype(str)

# drop rows with missing observations
merged_df = merged_df.dropna()

# generate column containing county
merged_df['county'] = merged_df['tract'].str[0:5]

merged_df.to_csv(path_to_final_dataset)
