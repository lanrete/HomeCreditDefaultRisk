#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/30

To work on the feature selection process.
Idea is to select one feature from multiple features aggregated from one source.
i.e, for amt_credit; we might create three different features for example
-- 1) amt_credit_sum
-- 2) amt_credit_max
-- 3) amt_credit_min
These three paramaters contains similar information, try and found the one with hightest
predictive power and exclude the others.
"""

# %%
import os
import gc
import numpy as np
import pandas as pd

from source.pipeline_utility.general_transformer import ColumnTypeExtract
from source.bureau_data_prepare import agg_bureau
from source.utility import timer

# %%
os.chdir('./source')
data_path = '../data'
train_base = pd.read_csv(f'{data_path}/application_train.csv')
test_base = pd.read_csv(f'{data_path}/application_test.csv')

train_base.set_index(keys='SK_ID_CURR', drop=True, inplace=True)
test_base.set_index(keys='SK_ID_CURR', drop=True, inplace=True)

with timer('Creating variables in base set'):
    for df in [train_base, test_base]:
        df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']

with timer('Aggregating bureau.csv'):
    bureau_df = agg_bureau()
    train_base = train_base.join(bureau_df, how='left')
    test_base = test_base.join(bureau_df, how='left')
    del bureau_df
    gc.collect()

# %%
from sklearn.preprocessing import LabelEncoder

y = train_base['TARGET']
del train_base['TARGET']
y = LabelEncoder().fit_transform(y)
x = train_base

del train_base
gc.collect()

# %%
extract_numerical = ColumnTypeExtract('Numerical')
num_x = extract_numerical.fit_transform(x)

# %%
col_list = num_x.columns
col_suffix = ['_'.join(column.split('_')[:-1]) for column in col_list]
col_df = pd.DataFrame({
    'Column': col_list,
    'Suffix': col_suffix,
})
col_df['Suffix'].value_counts()

# %%
col_cnt = col_df['Suffix'].value_counts()
col_df['Cnt'] = col_df['Suffix'].apply(
    lambda _: col_cnt[_]
)
col_df['Agg_Column'] = col_df['Column'].apply(
    lambda _: _.startswith('_')
)

# %%
filter_df = col_df[(col_df['Cnt'] > 1) & (col_df['Agg_Column'])].sort_values(
    by=['Cnt', 'Suffix'],
    ascending=[False, False]
)
print(f'Choose {len(filter_df.Suffix.drop_duplicates())} columns from {len(filter_df)} columns.')

# %%
for group in filter_df['Suffix'].drop_duplicates():
    column_list = filter_df.loc[filter_df['Suffix'] == group, 'Column']

# %%
from sklearn.feature_selection import f_classif, mutual_info_classif

temp_df = num_x[column_list]

# %%
calc_f = f_classif(temp_df, y)

# %%
temp_df.isnull().sum()

# %%
temp_df.notnull().sum()

# %%

# %%

