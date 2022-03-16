from datetime import datetime
import pandas as pd

"""
This file loads Eviction Lab's census tract-month level dataset of eviction filings. 
"""
evictions_df = pd.read_csv("https://evictionlab.org/uploads/all_sites_monthly_2020_2021.csv")

# our unemployment data is at the census tract level, so we drop zip code observations
evictions_df.drop(evictions_df.loc[evictions_df['type'] == "Zip Code"].index, inplace=True)

# certain records have missing/incorrect addresses; drop them
evictions_df.drop(evictions_df.loc[evictions_df['GEOID'] == "sealed"].index, inplace=True)

# GEOID is simply FIPS code for rows coresponding to census tracts
evictions_df.rename(columns={"GEOID": "fips"}, inplace=True)

# separate month into month and year
evictions_df['month'] = pd.to_datetime(evictions_df['month'])

# rename the column which contains filings data
evictions_df.rename(columns={'filings_2020': 'filings'}, inplace=True)

# drop unneeded columns
evictions_df.drop(columns=['city',
                           'type',
                           'racial_majority',
                           'filings_avg',
                           'last_updated'],
                  inplace=True)

# drop rows from after the implementation of the eviction moratorium on 9/4/2020
evictions_df.drop(evictions_df.loc[evictions_df['month'] >= datetime(2020, 9, 1)].index, inplace=True)

# save locally
evictions_df.to_csv("/Users/arjunshanmugam/Documents/School/Brown/Semester6/ECON1680/project1/project1-arjun-shanmugam/cleaned_data/evictions.csv")
