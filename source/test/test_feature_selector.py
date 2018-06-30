#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Lanrete on 2018/6/22

Unit test on feature selector
"""

import pandas as pd

from pipeline_utility.feature_selector import ModelBasedFeatureSelector
from sklearn.datasets import make_classification, make_regression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

cla_x, cla_y = make_classification(n_features=100)
reg_x, reg_y = make_regression(n_features=100)

cla_x_df = pd.DataFrame(cla_x)
reg_x_df = pd.DataFrame(reg_x)


class TestModelBasedFeatureSelector(object):
    def test_sanity_clf(self):
        transformer = ModelBasedFeatureSelector(RandomForestClassifier(), number_features=20)
        filtered_x = transformer.fit_transform(cla_x, cla_y)
        _, feature_cnt = filtered_x.shape
        assert feature_cnt == 20

    def test_sanity_reg(self):
        transformer = ModelBasedFeatureSelector(RandomForestRegressor(), number_features=25)
        filtered_x = transformer.fit_transform(reg_x, reg_y)
        _, feature_cnt = filtered_x.shape
        assert feature_cnt == 25

    def test_df_input(self):
        transformer = ModelBasedFeatureSelector(RandomForestClassifier(), number_features=15)
        filtered_x = transformer.fit_transform(cla_x_df, cla_y)
        _, feature_cnt = filtered_x.shape
        assert feature_cnt == 15

    def test_df_output(self):
        transformer = ModelBasedFeatureSelector(RandomForestClassifier(), number_features=24)
        filtered_x = transformer.fit_transform(cla_x_df, cla_y)
        assert isinstance(filtered_x, pd.DataFrame)

    def test_pnt_input(self):
        transformer = ModelBasedFeatureSelector(RandomForestClassifier(), percent_features=0.78)
        filtered_x = transformer.fit_transform(cla_x, cla_y)
        _, feature_cnt = filtered_x.shape
        assert feature_cnt == 78


if __name__ == '__main__':
    pass
