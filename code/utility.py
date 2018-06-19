#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/19

Includes some basic utility functions
"""

import time
from contextlib import contextmanager

import pandas as pd


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print(f'{title} done in {time.time() - t0:.2f} seconds.')


def category_processing(x):
    """
    To one-hot the categorical variables in the given dataset.

    Parameters
    ----------
    x : pd.DataFrame
        DataFrame with categorical variables to one-hot
    Returns
    -------
    dummy_x : pd.DataFrame
        Transformed DataFrame with dummy variables created for all categorical columns and object columns.

    category_columns: list
        List of newly-generated column names for dummy variables
    """
    original_columns = x.columns
    category_columns = x.select_dtypes(include=['object', 'category']).columns
    dummy_x = pd.get_dummies(x, columns=category_columns, dummy_na=True)
    category_columns = [_ for _ in dummy_x.columns if _ not in original_columns]
    return dummy_x, category_columns


if __name__ == '__main__':
    pass
