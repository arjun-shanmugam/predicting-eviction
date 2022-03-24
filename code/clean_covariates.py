import pandas as pd


"""
This file cleans Opportunity Insights' dataset of neighborhood characteristics by census tract.
"""
path_to_oi_data = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/raw_data/tract_covariates.csv"
path_to_clean_data = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/cleaned_data/covariates.csv"
oi_df = pd.read_csv(path_to_oi_data)

# generate a county_geoid variable which matches the other datasets
oi_df['state'] = oi_df['state'].astype(int).astype(str).str.zfill(2)
oi_df['county'] = oi_df['county'].astype(int).astype(str).str.zfill(3)
oi_df['tract'] = oi_df['tract'].astype(str).str.zfill(6)
oi_df['tract'] = oi_df['state'] + oi_df['county'] + oi_df['tract']

# drop unneeded columns
oi_df = oi_df.drop(columns=['state', 'county'])

# save locally
oi_df.to_csv("/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/cleaned_data/covariates.csv")
