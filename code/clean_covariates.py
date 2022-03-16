import pandas as pd

"""
This file cleans Opportunity Insights' dataset of neighborhood characteristics by census tract.
"""
# download file hosted on Google Drive
url = 'https://drive.google.com/file/d/1kdWE0S1OBE-iETQAOnaHRJwlFOaWezsT/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
oi_df = pd.read_csv(path)

# generate a county_geoid variable which matches the other datasets
oi_df['state'] = oi_df['state'].astype(int).astype(str).str.zfill(2)
oi_df['county'] = oi_df['county'].astype(int).astype(str).str.zfill(3)
oi_df['tract'] = oi_df['tract'].astype(str).str.zfill(6)
oi_df['fips'] = oi_df['state'] + oi_df['county'] + oi_df['tract']

# drop unneeded columns
oi_df.drop(columns=['state', 'county', 'tract'], inplace=True)

# save locally
oi_df.to_csv("/Users/arjunshanmugam/Documents/School/Brown/Semester6/ECON1680/project1/project1-arjun-shanmugam/cleaned_data/covariates.csv")
