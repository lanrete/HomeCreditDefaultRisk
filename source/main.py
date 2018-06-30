#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/15
"""

# TODO  Add previous_application features when ready

import gc
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from config import FIT_PARAMS
from utility import timer
from bureau_data_prepare import agg_bureau
from pre_application_data_prepare import agg_pre_application
from pipeline import fit_pipeline

DATA_PATH = '../data'


def main(fit_params):
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

    with timer('Aggregating previous_application.csv'):
        previous_application_df = agg_pre_application()
        train_base = train_base.join(previous_application_df, how='left')
        test_base = test_base.join(previous_application_df, how='left')
        del previous_application_df
        gc.collect()

    y = train_base['TARGET']
    del train_base['TARGET']
    y = LabelEncoder().fit_transform(y)

    header = 'Grid Searching Pipeline with parameter grids' if fit_params else 'Fitting and predicting'

    with timer(header):
        fit_pipeline(
            train_base, y,
            predict=True, x_score=test_base, fit_params=fit_params
        )


if __name__ == '__main__':
    main(fit_params=FIT_PARAMS)
