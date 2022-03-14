"""
This file builds the dataset used by my project.
"""
import numpy as np
import pandas as pd

"""CLEAN EVICTIONS DATA"""
evictions_df = pd.read_csv("https://eviction-lab-data-downloads.s3.amazonaws.com/ets/all_sites_weekly_2020_2021.csv")

# we cannot aggregate zip code-level observations to county level, so drop them
evictions_df.drop(evictions_df.loc[evictions_df['type'] == "Zip Code"].index, inplace=True)

# certain records have missing/incorrect addresses; drop them
evictions_df.drop(evictions_df.loc[evictions_df['GEOID'] == "sealed"].index, inplace=True)

# the numbers in indices 0-5 are the county code; store in separate column
evictions_df.rename(columns={"GEOID": "census_tract_geoid"}, inplace=True)
evictions_df['county_geoid'] = evictions_df['census_tract_geoid'].str.slice(start=0, stop=5)

# separate week_date into start_week_day, start_week_month, and start_week_year
evictions_df['day'] = pd.to_datetime(evictions_df['week_date']).dt.day
evictions_df['month'] = pd.to_datetime(evictions_df['week_date']).dt.month
evictions_df['year'] = pd.to_datetime(evictions_df['week_date']).dt.year

# drop unneeded columns
evictions_df.drop(columns=['city',
                           'type',
                           'census_tract_geoid',
                           'racial_majority',
                           'filings_avg',
                           'last_updated',
                           'week',
                           'week_date'],
                  inplace=True)

# rename the column which contains filings data
evictions_df.rename(columns={'filings_2020': 'filings'}, inplace=True)

"""CLEAN UNEMPLOYMENT DATA"""
laus_df = pd.read_csv('https://download.bls.gov/pub/time.series/la/la.data.64.County',
                      sep='\t',
                      names=['series_id', 'year', 'period', 'value', 'footnote_code'],
                      skiprows=1)

# separate series_id into its components
laus_df['geographic_level'] = laus_df['series_id'].str.slice(start=3, stop=5)
laus_df['adjusted_for_seasonality'] = np.where(laus_df['series_id'].str.slice(start=2, stop=2) == "S", 1, 0)
laus_df['county_geoid'] = laus_df['series_id'].str.slice(start=5, stop=10)
laus_df['measure_type'] = laus_df['series_id'].str.slice(start=18, stop=20)

# keep only observations which give data on unemployment rate
laus_df.drop(laus_df.loc[laus_df['measure_type'] != "03"].index, inplace=True)

# drop unneeded columns
laus_df.drop(columns=['geographic_level', 'adjusted_for_seasonality', 'footnote_code', 'series_id', 'measure_type'],
             inplace=True)

# replace M01, M02,... with 1, 2, ...
laus_df.replace(['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12', 'M13'],
                range(1, 14),
                inplace=True)
laus_df.drop(laus_df.loc[laus_df['period'] == 13].index, inplace=True)  # obs. w/ period=13 are year averages
laus_df.rename(columns={'period':'month', 'value':'unemployment_rate'})

print(laus_df.columns)
print(evictions_df.columns)

"""CLEAN COUNTY CHARACTERISTICS DATA"""
oi_df = pd.read_stata("https://drive.google.com/u/0/uc?id=1VGeRWTfT9Ir39Xe_wVNJtZhQQNfwyHJO&export=download")

# generate a county_geoid variable which matches the other datasets
oi_df['state'] = oi_df['state'].astype(int).astype(str).str.zfill(2)
oi_df['county'] = oi_df['county'].astype(int).astype(str).str.zfill(3)
oi_df['county_geoid'] = oi_df['state'] + oi_df['county']

# drop unneeded columns
oi_df.drop(columns=['state', 'county'], inplace=True)

"""MERGE DATASETS"""
# convert the column on which we want to join to a string across the board
evictions_df['county_geoid'] = evictions_df['county_geoid'].astype(str)
laus_df['county_geoid'] = laus_df['county_geoid'].astype(str)
oi_df['county_geoid'] = oi_df['county_geoid'].astype(str)

# join on county
evictions_and_laus = evictions_df.merge(laus_df, how='inner', on='county_geoid', validate='many_to_many')
merged_data = evictions_and_laus.merge(oi_df, how='inner', on='county_geoid', validate='many_to_one')
merged_data.to_csv("/Users/arjunshanmugam/Desktop")
# print(evictions_and_laus.head())
