import pandas as pd
import numpy as np


"""SWITCH THIS TO CENSUS TRACT LEVEL"""
oi_df = pd.read_stata("https://drive.google.com/u/0/uc?id=1VGeRWTfT9Ir39Xe_wVNJtZhQQNfwyHJO&export=download")

# generate a county_geoid variable which matches the other datasets
oi_df['state'] = oi_df['state'].astype(int).astype(str).str.zfill(2)
oi_df['county'] = oi_df['county'].astype(int).astype(str).str.zfill(3)
oi_df['county_geoid'] = oi_df['state'] + oi_df['county']

# drop unneeded columns
oi_df.drop(columns=['state', 'county'], inplace=True)


oi_df.to_csv("/Users/arjunshanmugam/Documents/School/Brown/Semester6/ECON1680/project1/project1-arjun-shanmugam/cleaned_data/covariates.csv")
