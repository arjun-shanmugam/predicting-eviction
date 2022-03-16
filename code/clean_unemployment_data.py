import numpy as np
import pandas as pd


"""
This file cleans DEEP-MAPS's census tract-month level unemployment dataset.
"""
# download file hosted on Google Drive
url = "https://drive.google.com/file/d/1ygzUi_MfCpLZh8TsLQnUIO75jkvjBngf/view?usp=sharing"
path = "https://drive.google.com/uc?export=download&id="+url.split('/')[-2]
unemployment_df = pd.read_csv(path)

