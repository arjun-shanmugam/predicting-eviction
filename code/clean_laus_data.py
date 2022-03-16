import numpy as np
import pandas as pd

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
laus_df.rename(columns={'period': 'month', 'value':'unemployment_rate'}, inplace=True)

print(laus_df.columns)
laus_df.to_csv("/Users/arjunshanmugam/Documents/School/Brown/Semester6/ECON1680/project1/project1-arjun-shanmugam/cleaned_data/laus.csv")
