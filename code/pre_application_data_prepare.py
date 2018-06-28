#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/23

To summarize the information in previous_application.csv
Aggregate into SK_ID_CURR level
"""

import gc

import numpy as np
import pandas as pd

from utility import timer
from utility import category_processing


def pre_app_clean_up(df: pd.DataFrame):
    # Changing XAP, XNA to np.nan
    df.replace(to_replace=['XAP', 'XNA'], value=[np.nan, np.nan], inplace=True)
    # This column is categorical data but coded as number
    df['SELLERPLACE_AREA'].replace(to_replace=-1, value=np.nan, inplace=True)
    df['SELLERPLACE_AREA'] = df['SELLERPLACE_AREA'].astype('str')
    # Mapping WEEKDAY to workday & weekend
    df['WEEKDAY_APPR_PROCESS_START'] = df['WEEKDAY_APPR_PROCESS_START'].replace(
        to_replace=['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY'],
        value='WEEKDAY',
    ).replace(
        to_replace=['SATURDAY', 'SUNDAY'],
        value='WEEKEND',
    )
    # For columns with date information, 365243 are considered as missing value
    for column in [
        'DAYS_FIRST_DRAWING',
        'DAYS_FIRST_DUE',
        'DAYS_LAST_DUE_1ST_VERSION',
        'DAYS_LAST_DUE',
        'DAYS_TERMINATION',
    ]:
        df[column].replace(to_replace=365243, value=np.nan, inplace=True)
    # change numerical -> flag
    df['NFLAG_LAST_APPL_IN_DAY'].replace(to_replace=[1, 0], value=['Y', 'N'], inplace=True)
    df['NFLAG_INSURED_ON_APPROVAL'].replace(to_replace=[1, 0], value=['Y', 'N'], inplace=True)
    # only keep last application per contract
    ret_df = df[df['FLAG_LAST_APPL_PER_CONTRACT'] == 'Y'].copy()
    ret_df.drop(columns=[
        'FLAG_LAST_APPL_PER_CONTRACT',
        'NAME_CLIENT_TYPE',
        'PRODUCT_COMBINATION',
        'SELLERPLACE_AREA',
    ], inplace=True)

    del df
    gc.collect()

    return ret_df


def agg_pre_application():
    """
        aggregation function to apply on each group of 'SK_ID_CURR'
    Returns
    -------
    agg_df : pd.DataFrame
        Aggregated DataFrame
    """
    previous_application = pd.read_csv('../data/previous_application.csv')
    print(f'|--previous_application.csv ==> {previous_application.shape}')

    print('|--Cleaning up the dataset...')
    previous_application = pre_app_clean_up(previous_application)

    print('|--Making dummies for categorical data...')
    previous_application, previous_application_cc = category_processing(previous_application)

    # For numerical columns, set up the basic aggregation function
    numerical_dict = {
        'SK_ID_PREV': [len],

        'AMT_ANNUITY': ['max', 'min', 'mean', 'sum'],
        'AMT_APPLICATION': ['max', 'min', 'mean', 'sum'],
        'AMT_CREDIT': ['max', 'min', 'mean'],
        'AMT_DOWN_PAYMENT': ['max', 'min', 'sum'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],

        'HOUR_APPR_PROCESS_START': ['max', 'min', 'mean'],

        # 'RATE_DOWN_PAYMENT': ['max', 'min', 'mean'],
        # 'RATE_INTEREST_PRIMARY': ['max', 'min', 'mean'],
        # 'RATE_INTEREST_PRIVILEGED': ['max', 'min', 'mean'],

        'DAYS_DECISION': ['max', 'min', 'mean'],
        # 'DAYS_FIRST_DRAWING': ['max', 'min', 'mean'],
        # 'DAYS_FIRST_DUE': ['max', 'min', 'mean'],
        # 'DAYS_LAST_DUE_1ST_VERSION': ['max', 'min', 'mean'],
        # 'DAYS_LAST_DUE': ['max', 'min', 'mean'],
        'DAYS_TERMINATION': ['max', 'min', 'mean'],

        'CNT_PAYMENT': ['max', 'min', 'mean'],
    }

    # For categorical columns, adding up the observation
    categorical_dict = {dummy_column: ['sum'] for dummy_column in previous_application_cc}

    print('|--Aggregating features on whole data...')
    agg_df = previous_application.groupby(by=['SK_ID_CURR']).agg({**numerical_dict, **categorical_dict})
    agg_df.columns = pd.Index(
        [f'PRE_APP_{e[0]}_{e[1].upper()}' for e in agg_df.columns.tolist()]
    )

    print('|--Aggregating features for approved contract...')
    approved_df = previous_application[previous_application['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg_df = approved_df.groupby(by=['SK_ID_CURR']).agg(numerical_dict)
    approved_agg_df.columns = pd.Index(
        [f'PRE_APP_Approved_{e[0]}_{e[1].upper()}' for e in approved_agg_df.columns.tolist()]
    )

    print('|--Aggregating featrures for rejected contract...')
    rejected_df = previous_application[previous_application['NAME_CONTRACT_STATUS_Approved'] == 0]
    rejected_agg_df = rejected_df.groupby(by=['SK_ID_CURR']).agg(numerical_dict)
    rejected_agg_df.columns = pd.Index(
        [f'PRE_APP_Rejected_{e[0]}_{e[1].upper()}' for e in rejected_agg_df.columns.tolist()]
    )

    agg_df = agg_df.join(approved_agg_df, how='left')
    agg_df = agg_df.join(rejected_agg_df, how='left')

    return agg_df


if __name__ == '__main__':
    with timer('Aggregating previous_application.csv'):
        previous_application_agg_df = agg_pre_application()
        print('|--Saving the aggregated files...')
        previous_application_agg_df.to_csv('../data/_agg_pre_app.csv')
