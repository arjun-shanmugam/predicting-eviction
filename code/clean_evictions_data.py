from datetime import datetime
import pandas as pd


"""
This file loads Eviction Lab's census tract-month level dataset of eviction filings. 
"""
path_to_evictions_data = "/Users/arjunshanmugam/Documents/School/Brown/Semester6/ECON1680/project1/project1-arjun-shanmugam/raw_data/all_sites_monthly_2020_2021.csv"
evictions_df = pd.read_csv(path_to_evictions_data)

# our unemployment data is at the census tract level, so we drop zip code observations
evictions_df = evictions_df.drop(evictions_df.loc[evictions_df['type'] == "Zip Code"].index)

# certain records have missing/incorrect addresses; drop them
evictions_df = evictions_df.drop(evictions_df.loc[evictions_df['GEOID'] == "sealed"].index)

# GEOID is simply FIPS code for rows corresponding to census tracts
evictions_df = evictions_df.rename(columns={"GEOID": "fips"})

# separate month into month and year
evictions_df['month'] = pd.to_datetime(evictions_df['month'])

# rename the column which contains filings data
evictions_df = evictions_df.rename(columns={'filings_2020': 'filings'})

# drop unneeded columns
evictions_df = evictions_df.drop(columns=['city',
                           'type',
                           'racial_majority',
                           'filings_avg',
                           'last_updated'])

# drop rows from after the implementation of the eviction moratorium on 9/4/2020
evictions_df = evictions_df.drop(evictions_df.loc[evictions_df['month'] >= datetime(2020, 9, 1)].index)

# save locally
evictions_df.to_csv("/Users/arjunshanmugam/Documents/School/Brown/Semester6/ECON1680/project1/project1-arjun-shanmugam/cleaned_data/evictions.csv")
