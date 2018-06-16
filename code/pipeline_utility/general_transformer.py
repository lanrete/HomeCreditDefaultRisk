#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/16

Some general transformers
Including
    1) Extract column by names
    2) Extract categorical columns/numerical columns
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnExtract(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        if isinstance(columns, list):
            self.columns = columns
        else:
            self.columns = [columns]

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **trans_params):
        return X[self.columns].copy()


class ColumnTypeExtract(BaseEstimator, TransformerMixin):
    def __init__(self, types):
        """
        Transformer to return only categorical or numerical columns
        :param types: 'Category' or 'Numerical'.
                      'Category': return DataFrame with all columns with string type and pd.Categorical type
                      'Numerical': return DataFrame with all columns with numerical type
        """
        if types not in ['Category', 'Numerical']:
            raise ValueError(f'types must be either Category or Numerical, input is {types}')
        self.types = types

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **trans_params):
        if self.types == 'Category':
            return X.select_dtypes(include=['object', 'category']).copy()
        if self.types == 'Numerical':
            return X.select_dtypes(exclude=['object', 'category']).copy()
        return X


class GeneralImputer(BaseEstimator, TransformerMixin):
    def __init__(self, value=None, func=None):
        """
        A general imputer to fill-in missing value.
        Pass a fixed-value or a callable method to get the missing impution
        :param value: the fixed value to impute all the missing information
                      ignored if func is passed
                      if a DataFrame with multiple columns is passed, this single value will be used to
                      impute every column with missing value
        :param func: the callable function to calculate the impution value with the non-missing part
                     if a DataFrame witth multiple columns is passed, the function will be applied to the
                     non-missing part of each column and get the impute values for each column separately.
        """
        if func is None:
            self.impute_type = 'fixed'
            self.value = value
        elif callable(func):
            self.impute_type = 'function'
            self.func = func
        else:
            raise TypeError('func should be a callable object/a function')
        self.value_dict = {}

    def fit(self, X, y=None, **fit_params):
        if self.impute_type == 'fixed':
            self.value_dict = {_: self.value for _ in X.columns}
        if self.impute_type == 'function':
            for _ in X.columns:
                non_missing_value = X[_][X[_].notnull()]
                impute_value = self.func(non_missing_value)
                self.value_dict[_] = impute_value
        return self

    def transform(self, X, y=None, **fit_parms):
        for _ in X.columns:
            X[_].fillna(self.value_dict[_], inplace=True)
        return X