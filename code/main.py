#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/15
"""

import gc
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utility import timer
from bureau_data_prepare import agg_bureau
from pipeline import fit_pipeline_parameters


DATA_PATH = '../data'


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
    x_train, x_test, y_train, y_test = train_test_split(train_base, y, test_size=0.33, random_state=2203)
    del y
    gc.collect()

    with timer('Grid Searching Pipeline with parameter grids'):
        fit_pipeline_parameters(
            x_train, y_train, x_test, y_test,
            predict=True, x_score=test_base, submission='with_bureau'
        )


if __name__ == '__main__':
    main()
