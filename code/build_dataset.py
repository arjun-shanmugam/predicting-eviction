"""
This file builds the dataset used by my project.
"""
import numpy as np
import pandas as pd

# read in evictions and LAUS data
evictions_df = pd.read_csv("/Users/arjunshanmugam/Documents/School/Brown/Semester6/ECON1680/project1/project1-arjun-shanmugam/cleaned_data/evictions.csv")
laus_df = pd.read_csv("/Users/arjunshanmugam/Documents/School/Brown/Semester6/ECON1680/project1/project1-arjun-shanmugam/cleaned_data/laus.csv")

# join on county
evictions_and_laus = evictions_df.merge(laus_df, how='inner', on=['county_geoid', 'year', 'month'])

# read in OI data and join on county
oi_df = pd.read_csv("/Users/arjunshanmugam/Documents/School/Brown/Semester6/ECON1680/project1/project1-arjun-shanmugam/cleaned_data/covariates.csv")
merged_data = evictions_and_laus.merge(oi_df, how='inner', on='county_geoid', validate='many_to_one')
merged_data.to_csv("/Users/arjunshanmugam/Documents/School/Brown/Semester6/ECON1680/project1/project1-arjun-shanmugam/cleaned_data/merged_data.csv")
