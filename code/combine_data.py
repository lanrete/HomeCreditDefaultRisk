#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/19

To combine data sources (base and aggregated one) into the combined data source for final pipeline
"""

import os
import glob
import pandas as pd


def combine_data():
    train_base = pd.read_csv('./data/application_train.csv')
    test_base = pd.read_csv('./data/application_test.csv')
    train_base.set_index(keys='SK_ID_CURR', drop=True, inplace=True)
    test_base.set_index(keys='SK_ID_CURR', drop=True, inplace=True)
    print(f'Train base ==> {train_base.shape}')
    print(f'Test base ==> {test_base.shape}')
    file_list = glob.glob('./data/_agg_*.csv')
    for file in file_list:
        file_name = os.path.basename(file)
        temp_df = pd.read_csv(file)
        temp_df.set_index(keys='SK_ID_CURR', drop=True, inplace=True)
        train_base = train_base.join(temp_df)
        test_base = test_base.join(temp_df)
        print(f'Train base after joining with {file_name} ==> {train_base.shape}')
        print(f'Test base after joining with {file_name} ==> {test_base.shape}')
    train_base.to_csv('./data/_final_train.csv')
    test_base.to_csv('./data/_final_test.csv')


if __name__ == '__main__':
    combine_data()
