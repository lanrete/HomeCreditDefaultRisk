#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/27
"""
# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pre_application_data_prepare import pre_app_clean_up

plt.style.use('fivethirtyeight')

# %%
base = pd.read_csv('./data/application_train.csv')
previous = pd.read_csv('./data/previous_application.csv')

print(f'base ==> {base.shape}')
print(f'previous ==> {previous.shape}')

# %%
pre_app = previous['SK_ID_CURR'].value_counts()
id_w_most_pre_app = pre_app.index[0]
temp_df = previous[previous['SK_ID_CURR'] == id_w_most_pre_app]

# %%
_, ax = plt.subplots()
ax = ax  # type: plt.Axes
sns.countplot(x='NAME_CONTRACT_TYPE', data=temp_df, ax=ax)
ax.set_title('Contract Type')
plt.show()

# %%
_, ax = plt.subplots()
ax = ax  # type: plt.Axes
sns.countplot(x='NAME_CONTRACT_STATUS', data=temp_df, ax=ax)
ax.set_title('Contract Status')
plt.show()

# %% Seems like XNA equals missing here
temp_df.replace(to_replace='XNA', value=np.nan, inplace=True)

# %%
print(temp_df.loc[temp_df['NAME_CONTRACT_STATUS'] == 'Approved', 'NAME_PAYMENT_TYPE'].value_counts())
print(temp_df['NAME_PAYMENT_TYPE'].value_counts())

# %%
temp_df.groupby(by=['NAME_PAYMENT_TYPE', 'NAME_CONTRACT_STATUS', 'NAME_CONTRACT_TYPE']).agg(
    {'SK_ID_CURR': len}
)

# %%
temp_df.groupby(by=['NAME_CONTRACT_STATUS', 'CODE_REJECT_REASON']).agg(
    {'SK_ID_CURR': len}
)

# %%
previous.drop(columns=['WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'NAME_CLIENT_TYPE'], inplace=True)
previous['FLAG_LAST_APPL_PER_CONTRACT'].value_counts()

# %%
previous['NFLAG_LAST_APPL_IN_DAY'].value_counts()

# %%
for _ in ['NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 'NAME_GOODS_CATEGORY']:
    print(previous[_].value_counts())

# %%
sample_df = previous.iloc[0:1000, :]  # type:pd.DataFrame
print(f'Sample data generated ==> {sample_df.shape}')
raw_sawple_df = sample_df.copy()

# %%
sample_df = raw_sawple_df

sample_df = pre_app_clean_up(sample_df)

# %%
previous['NFLAG_LAST_APPL_IN_DAY'].value_counts()

# %%
