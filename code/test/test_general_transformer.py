#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/16

Unit test on base transformer
"""

import pytest

from pipeline_utility.general_transformer import ColumnExtract
from pipeline_utility.general_transformer import ColumnTypeExtract
from pipeline_utility.general_transformer import GeneralImputer

import pandas as pd
import numpy as np

from pandas.util.testing import assert_frame_equal


class TestColumnExtract(object):
    @pytest.fixture(scope='class')
    def sample_data(self):
        x_train = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [1, 3, np.nan],
        })
        x_test = pd.DataFrame({
            'A': [2, 3],
            'B': ['a', 'b'],
            'D': [1, 3],
        })
        return x_train, x_test

    def test_single_column_input(self, sample_data):
        x_train, x_test = sample_data
        extract = ColumnExtract('A')
        return_output = extract.fit_transform(x_train)
        assert isinstance(return_output, pd.DataFrame)

    def test_multi_column_input(self, sample_data):
        x_train, x_test = sample_data
        extract = ColumnExtract(['A', 'B'])
        return_output = extract.fit_transform(x_train)
        assert isinstance(return_output, pd.DataFrame)

    def test_fit_transform(self, sample_data):
        x_train, x_test = sample_data
        extract = ColumnExtract('A')
        train_extracted = extract.fit_transform(x_train)
        test_extracted = extract.transform(x_test)
        assert test_extracted.columns == train_extracted.columns


class TestColumnTypeExtract(object):
    @pytest.fixture(scope='class')
    def sample_data(self):
        x_train = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [1, 3, np.nan],
        })
        x_test = pd.DataFrame({
            'A': [2, 3],
            'B': ['a', 'b'],
            'C': [1, 3],
        })
        return x_train, x_test

    def test_return_numeric(self, sample_data):
        x_train, x_test = sample_data
        extract = ColumnTypeExtract('Numerical')
        train_return = extract.fit_transform(x_train)
        assert list(train_return.columns) == ['A', 'C']

    def test_return_category(self, sample_data):
        x_train, x_test = sample_data
        extract = ColumnTypeExtract('Category')
        train_return = extract.fit_transform(x_train)
        assert list(train_return.columns) == ['B']

    def test_return_test(self, sample_data):
        x_train, x_test = sample_data
        extract = ColumnTypeExtract('Numerical')
        train_return = extract.fit_transform(x_train)
        test_return = extract.transform(x_test)
        assert list(train_return.columns) == list(test_return.columns)

    def test_wrong_input(self):
        with pytest.raises(ValueError):
            ColumnTypeExtract('Nonsense')


class TestGeneralImputer(object):
    @pytest.fixture()
    def sample_data(self):
        x_train = pd.DataFrame({
            'A': [1, 2, np.nan, 3, 4],
            'B': [1, 2, 3, np.nan, np.nan]
        })

        x_test = pd.DataFrame({
            'A': [1, np.nan, np.nan, np.nan, np.nan],
            'B': [1, 2, 3, 4, 5]
        })
        return x_train, x_test

    def test_fixed_value_impute(self, sample_data):
        x_train, x_test = sample_data
        imputer = GeneralImputer(value=5)
        train_return = imputer.fit_transform(x_train)
        expected_df = pd.DataFrame({
            'A': [1, 2, 5, 3, 4],
            'B': [1, 2, 3, 5, 5],
        })
        assert_frame_equal(left=train_return, right=expected_df, check_dtype=False)

    def test_callable_impute(self, sample_data):
        x_train, x_test = sample_data
        imputer = GeneralImputer(func=np.nanmax)
        train_return = imputer.fit_transform(x_train)
        expected_df = pd.DataFrame({
            'A': [1, 2, 4, 3, 4],
            'B': [1, 2, 3, 3, 3]
        })
        assert_frame_equal(left=train_return, right=expected_df, check_dtype=False)

    def test_callable_impute_test(self, sample_data):
        x_train, x_test = sample_data
        imputer = GeneralImputer(func=np.nanmin)
        imputer.fit(x_train)
        test_return = imputer.transform(x_test)
        expected_df = pd.DataFrame({
            'A': [1, 1, 1, 1, 1],
            'B': [1, 2, 3, 4, 5]
        })
        assert_frame_equal(left=test_return, right=expected_df, check_dtype=False)


if __name__ == '__main__':
    pass
