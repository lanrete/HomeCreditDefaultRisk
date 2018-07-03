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

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import minmax_scale

processed = temp_df

processed = Imputer().fit_transform(processed)
# processed = minmax_scale(processed)

# %%
from sklearn.feature_selection import chi2, f_classif

chi2_score = chi2(processed, y)
f_score = f_classif(processed, y)

# %%
chi2_score

# %%
f_score


# %%
def select_feature(_x, _y):
    col_list = _x.columns
    _x = Imputer().fit_transform(_x)
    _x = minmax_scale(_x)
    score = f_classif(_x, _y)[0]
    return col_list[score.argmax()]


selected = select_feature(temp_df, y)

# %%
print(selected)

# %%
temp_df = x[filter_df['Column']]


# %%
def select_feature(_x, _y):
    col_list = _x.columns
    _x = Imputer().fit_transform(_x)
    _x = minmax_scale(_x)
    score = f_classif(_x, _y)[0]
    return score


score_list = select_feature(temp_df, y)

# %%
filter_df['Score'] = score_list
selected_features = filter_df.groupby(by=['Suffix']).apply(
    lambda d: d.loc[d['Score'].idxmax(), 'Column']
)

# %%
filter_df.loc[filter_df['Suffix'] == '_BUREAU_ACTIVE_AMT_CREDIT_MAX_OVERDUE']


# %%
def get_features_list(_x, _y):
    col_df = pd.DataFrame({
        'Column': _x.columns
    })
    col_df['Suffix'] = col_df['Column'].apply(
        lambda col: '_'.join(col.split('_')[:-1])
    )
    suffix_count = col_df['Suffix'].value_counts()
    col_df['Need Selection'] = col_df['Suffix'].apply(
        lambda suffix: suffix_count[suffix] > 1 and suffix.startswith('_')
    )
    selection_df = col_df[col_df['Need Selection']].copy()

    part_x = _x[selection_df['Column']]
    processed_part_x = Imputer().fit_transform(part_x)
    processed_part_x = minmax_scale(processed_part_x)
    score = f_classif(processed_part_x, _y)[0]
    selection_df['Score'] = score

    selected_features = selection_df.groupby(by=['Suffix']).apply(
        lambda part_df: part_df.loc[part_df['Score'].idxmax(), 'Column']
    )

    ret_list = list(selected_features)
    ret_list.extend(col_df.loc[~col_df['Need Selection'], 'Column'])
    return ret_list


selected_features = get_features_list(num_x, y)

# %%
len(selected_features)

# %%
no_need_df = col_df.loc[~col_df['Need Selection'], 'Column']

# %%
# TODO  Sumarize function get_features_list into transformer to be used in pipeline
