#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/21

Hold the transformers to select features within the pipeline
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class ModelBasedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    A feature selector to select features based on the `feature_importances_` from a fitted estimators.

    The x and y will be used to fit the given estimators once, the features will be sorted by the `feature_importances_`
    and top features will be kept as the transform result.
    """
    def __init__(self, clf, number_features=None, percent_features=None):
        """

        Parameters
        ----------
        clf : sklearn estimator
            The estimator used to calculate the `feature_importances_`on complete dataset.
            Estimator given must have attributes `feature_importances_` after fitted.
        number_features : int, default None
            Number of features to keep after feature selection, must be in (0, # total features]
        percent_features : float, default None
            Percentage of features to keep after feature selection, must be in (0, 1]
            If original dataset have 100 features and `percent_features=0.25`, then 25 features with
            highest `feature_importances_` will be selected
        """
        self.clf = clf
        self.selected_columns = None
        if number_features:
            assert number_features > 0
            self.number_features = number_features
            self.percent_features = None
            self.select_type = 'Number'
        elif percent_features:
            assert 0 < percent_features <= 1
            self.number_features = None
            self.percent_features = percent_features
            self.select_type = 'Percent'
        else:
            raise ValueError('Either number_features or percent_features should be given')

    def fit(self, X, y=None, **fit_params):
        _, total_columns = X.shape
        self.clf.fit(X, y, **fit_params)
        importance = self.clf.feature_importances_  # type: np.ndarray
        column_index = importance.argsort()

        if self.select_type == 'Percent':
            self.number_features = int(total_columns * self.percent_features)
        if self.number_features > total_columns:
            self.number_features = total_columns
            raise RuntimeWarning(f'{self.number_features} > {total_columns} which is the total features available')

        self.selected_columns = column_index[-self.number_features:]
        return self

    def transform(self, X, y=None, **transform_params):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_columns]
        else:
            return X[:, self.selected_columns]
