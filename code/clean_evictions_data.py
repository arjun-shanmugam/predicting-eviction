import pandas as pd
import numpy as np

evictions_df = pd.read_csv("https://evictionlab.org/uploads/all_sites_monthly_2020_2021.csv")

# our unemployment data is at the census tract level, so we drop zip code observations
evictions_df.drop(evictions_df.loc[evictions_df['type'] == "Zip Code"].index, inplace=True)

# certain records have missing/incorrect addresses; drop them
evictions_df.drop(evictions_df.loc[evictions_df['GEOID'] == "sealed"].index, inplace=True)

# GEOID is simply FIPS code for rows coresponding to census tracts
evictions_df.rename(columns={"GEOID": "fips"}, inplace=True)

# separate month into month and year
evictions_df['month'] = pd.to_datetime(evictions_df['month']).dt.month
evictions_df['year'] = pd.to_datetime(evictions_df['month']).dt.year

# rename the column which contains filings data
evictions_df.rename(columns={'filings_2020': 'filings'}, inplace=True)

# drop unneeded columns
evictions_df.drop(columns=['city',
                           'type',
                           'racial_majority',
                           'filings_avg',
                           'last_updated'],
                  inplace=True)

evictions_df.to_csv("/Users/arjunshanmugam/Documents/School/Brown/Semester6/ECON1680/project1/project1-arjun-shanmugam/cleaned_data/evictions.csv")
