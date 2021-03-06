#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/19

To summarize the information in bureau.csv
Aggregate into SK_ID_CURR level

"""

import numpy as np
import pandas as pd

from utility import timer
from utility import category_processing


def bureau_clean_up(df: pd.DataFrame):
    df.replace(to_replace=['XAP', 'XNA'], value=[np.nan, np.nan], inplace=True)
    for col in df.columns:
        if col.startswith('DAYS_'):
            df[col].replace(to_replace=365243, value=np.nan, inplace=True)
    return df


def agg_bureau():
    """
    aggregation function to apply on each group of 'SK_ID_CURR'
    Returns
    -------
    agg_df : pd.DataFrame
        Aggregated DataFrame
    """
    bureau = pd.read_csv('../data/bureau.csv')
    print(f'|--bureau.csv ==> {bureau.shape}')

    print('|--Cleaning up the dataset...')
    bureau = bureau_clean_up(bureau)

    print('|--Making dummies for categorical data...')
    bureau, bureau_cc = category_processing(bureau)
    # For numerical columns, set up the basic aggregation function

    numerical_dict = {
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],

        'DAYS_CREDIT': ['max', 'min', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'median'],
        'DAYS_CREDIT_UPDATE': ['mean'],

        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean', 'median', 'min'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_LIMIT': ['max', 'mean', 'min'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'sum', 'mean'],
        'AMT_ANNUITY': ['sum'],

        'CNT_CREDIT_PROLONG': ['sum', 'mean'],
    }

    # For categorical columns, adding up the observation
    categorical_dict = {dummy_column: ['sum'] for dummy_column in bureau_cc}

    print('|--Aggregating features on whole data...')
    agg_df = bureau.groupby(by='SK_ID_CURR').agg({**numerical_dict, **categorical_dict})
    agg_df.columns = pd.Index(
        [f'_BUREAU_{e[0]}_{e[1].upper()}' for e in agg_df.columns.tolist()]
    )

    print('|--Aggregating features for active credit...')
    active_df = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg_df = active_df.groupby(by='SK_ID_CURR').agg(numerical_dict)
    active_agg_df.columns = pd.Index(
        [f'_BUREAU_ACTIVE_{e[0]}_{e[1].upper()}' for e in active_agg_df.columns.tolist()]
    )

    print('|--Aggregating featrures for closed credit...')
    closed_df = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg_df = closed_df.groupby(by='SK_ID_CURR').agg(numerical_dict)
    closed_agg_df.columns = pd.Index(
        [f'_BUREAU_CLOSED_{e[0]}_{e[1].upper()}' for e in closed_agg_df.columns.tolist()]
    )

    agg_df = agg_df.join(active_agg_df, how='left')
    agg_df = agg_df.join(closed_agg_df, how='left')

    return agg_df


if __name__ == '__main__':
    with timer('Aggregating bureau.csv'):
        bureau_agg_df = agg_bureau()
        print('|--Saving the aggregated files...')
        bureau_agg_df.to_csv('../data/_agg_bureau.csv')
