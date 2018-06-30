#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/7/1
"""
#%%
import numpy as np
import pandas as pd

#%%
bureau = pd.read_csv('./data/bureau.csv')
print(f'bureau.csv ==> {bureau.shape}')

#%%
replace_df = bureau.replace(to_replace=['XNA', 'XAP'], value=np.nan)
from pandas.testing import assert_frame_equal
assert_frame_equal(replace_df, bureau)


#%%
for col in bureau.columns:
    if col.startswith('DAYS_'):
        print(col)
        print(bureau[col].max())
        print(bureau[col].min())
        replace_df[col].replace(to_replace=365243, value=np.nan, inplace=True)

#%%
assert_frame_equal(replace_df, bureau)

#%%