#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/15
"""

import gc
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from utility import timer
from bureau_data_prepare import agg_bureau
from pipeline import fit_pipeline


DATA_PATH = '../data'
FIT_PARAMS = False


def main():
    data_path = '../data'
    train_base = pd.read_csv(f'{data_path}/application_train.csv')
    test_base = pd.read_csv(f'{data_path}/application_test.csv')

    train_base.set_index(keys='SK_ID_CURR', drop=True, inplace=True)
    test_base.set_index(keys='SK_ID_CURR', drop=True, inplace=True)

    with timer('Aggregating bureau.csv'):
        bureau_df = agg_bureau()
        train_base = train_base.join(bureau_df, how='left')
        test_base = test_base.join(bureau_df, how='left')
        del bureau_df
        gc.collect()

    y = train_base['TARGET']
    del train_base['TARGET']
    y = LabelEncoder().fit_transform(y)

    header = 'Grid Searching Pipeline with parameter grids' if FIT_PARAMS else 'Fitting and predicting'

    with timer(header):
        fit_pipeline(
            train_base, y,
            predict=True, x_score=test_base, fit_params=FIT_PARAMS
        )


if __name__ == '__main__':
    main()
