import pandas as pd


"""
This file cleans DEEP-MAPS's census tract-month level unemployment dataset.
"""
path_to_deep_maps_data = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/raw_data/deepmaps_tractdata_december2020_prelim.csv"
path_to_clean_data = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/cleaned_data/unemployment.csv"
unemployment_df = pd.read_csv(path_to_deep_maps_data)

# drop observations corresponding to unemployment statistics for specific subsets of the population
unemployment_df = unemployment_df.drop(unemployment_df.loc[unemployment_df['cat'] != 'total'].index)
unemployment_df = unemployment_df.drop(unemployment_df.loc[unemployment_df['grp'] != 'total'].index)

# drop columns indicating groups to which estimates correspond
unemployment_df = unemployment_df.drop(columns=['cat', 'grp'])

# convert data from wide to long format
unemployment_df = pd.wide_to_long(unemployment_df,
                                  ['cnip_2020', 'laborforce_2020', 'employed_2020'],
                                  i='fips',
                                  j='month',
                                  sep='_')

# wide_to_long sets index as (i, j); reset index so that i, j are columns
unemployment_df = unemployment_df.reset_index()

# reformat the month column to be of form MM/2020 and turn it into a datetime obj
unemployment_df['month'] = unemployment_df['month'].astype(str).str.zfill(2)
unemployment_df['month'] = unemployment_df['month'] + "/2020"
unemployment_df['month'] = pd.to_datetime(unemployment_df['month'])

# create unemployment rate column and drop employment, labor force columns
unemployment_df['unemployment_rate'] = 1 - unemployment_df['employed_2020'] / unemployment_df['laborforce_2020']
unemployment_df = unemployment_df.drop(columns=['employed_2020', 'laborforce_2020'])

# rename CNIP colum
unemployment_df = unemployment_df.rename(columns={'cnip_2020': 'cnip'})

# rename fips to tract
unemployment_df = unemployment_df.rename(columns={'fips': 'tract'})
print(unemployment_df.columns)

# save data
unemployment_df.to_csv(path_to_clean_data)