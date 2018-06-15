#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/15
"""

import os
import glob
import numpy as np
import pandas as pd


DATA_PATH = './data'


if __name__ == '__main__':
    data_list = glob.glob(f'{DATA_PATH}/*.csv')
    df_dict = {}
    for data in data_list:
        data_name = os.path.split(data)[-1]
        if data_name in ['HomeCredit_columns_description.csv', 'sample_submission.csv']:
            continue
        df_dict[data_name] = pd.read_csv(data, encoding='utf-8')
        print(f'{data_name} ==> {df_dict[data_name].shape}')
